"""
============================================================
Assignment 5 — Checkpoint 2
Task 3 (pre-training similarity) + Task 4 (SimCLR components)
============================================================
Implements and tests:
  • Encoder  (ResNet-18 modified for CIFAR-10)
  • ProjectionHead  (512 → 256 → 128)
  • SimCLR model combining both
  • Positive / negative pair indexing
  • Cosine similarity matrix
  • NT-Xent contrastive loss  (implemented from scratch)
  • Similarity heatmap visualisation

Outputs
-------
results/similarity_matrix_before_training.png
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset, TwoViewDataset
from torch.utils.data import DataLoader
from rollNumber_05_task2_augmentations import TwoViewTransform, simclr_transform

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 2026
BATCH_SIZE  = 64
TAU         = 0.5          # temperature for NT-Xent
PROJ_DIM    = 128
DATA_ROOT   = "./data"
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Encoder
# ─────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    ResNet-18 modified for CIFAR-10.

    Changes from the default torchvision ResNet-18
    -----------------------------------------------
    • conv1  : 7×7 stride-2 → 3×3 stride-1, padding 1
    • maxpool: replaced with nn.Identity (no aggressive downsampling)
    • fc     : removed / replaced with Identity to expose 512-dim features

    The encoder produces a 512-dimensional representation h for each image.
    """

    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)

        # ── CIFAR-10 modifications ────────────────────────────────────────────
        backbone.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                                     stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        # Keep every layer except the original classification head
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,   # Identity — keeps the interface clean
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,   # output: (B, 512, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, 512) feature vectors."""
        h = self.features(x)
        return h.flatten(1)       # (B, 512)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Projection Head
# ─────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    MLP projection head used ONLY during SimCLR pre-training.

    Architecture (as specified in the assignment):
        Linear(512 → 256)  →  ReLU  →  Linear(256 → 128)

    The output z is what enters the NT-Xent loss.
    The encoder features h (not z) are used for downstream tasks.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns (B, 128) projected representations."""
        return self.net(h)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SimCLR model
# ─────────────────────────────────────────────────────────────────────────────
class SimCLR(nn.Module):
    """
    Full SimCLR model = Encoder + Projection Head.
    Takes two views and returns their projected representations z1, z2.
    """

    def __init__(self):
        super().__init__()
        self.encoder    = Encoder()
        self.projection = ProjectionHead()

    def forward(self, view1: torch.Tensor, view2: torch.Tensor):
        """
        Args:
            view1: (N, 3, 32, 32)
            view2: (N, 3, 32, 32)
        Returns:
            z1, z2: each (N, 128)  — projected representations
        """
        h1 = self.encoder(view1)    # (N, 512)
        h2 = self.encoder(view2)    # (N, 512)
        z1 = self.projection(h1)    # (N, 128)
        z2 = self.projection(h2)    # (N, 128)
        return z1, z2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Positive / Negative pair construction
# ─────────────────────────────────────────────────────────────────────────────
def describe_pairs(N: int = 4) -> None:
    """
    Print the pair table for a batch of N original images.
    For a batch of N originals, we have 2N views:
        indices 0 … N-1   → view 1 of each image
        indices N … 2N-1  → view 2 of each image
    """
    print("\nPositive / Negative pair construction")
    print(f"  Batch size (original images) : {N}")
    print(f"  Total views                  : {2 * N}  (view-1 indices 0–{N-1}, "
          f"view-2 indices {N}–{2*N-1})\n")
    print(f"  {'Original Image':<20} {'View 1 Index':>14} {'View 2 Index':>14} {'Positive Pair':>14}")
    print("  " + "-" * 64)
    for i in range(N):
        print(f"  {'image ' + str(i):<20} {i:>14} {i + N:>14} {'yes':>14}")
    print()
    print("  All other (view_i, view_j) pairs where j ≠ i and j ≠ i+N")
    print("  are treated as NEGATIVE pairs.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cosine Similarity Matrix
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity_matrix(z1: torch.Tensor,
                              z2: torch.Tensor) -> torch.Tensor:
    """
    Build the full (2N × 2N) cosine similarity matrix.

    Steps
    -----
    1. Concatenate z1 and z2 into Z of shape (2N, D).
    2. L2-normalise each row so dot product = cosine similarity.
    3. Return Z @ Z.T.

    The diagonal holds sim(z_i, z_i) = 1 and is excluded from the loss.
    Positive pairs are at positions (i, i+N) and (i+N, i) for i in [0, N).
    Every other entry is a negative pair.
    """
    Z = torch.cat([z1, z2], dim=0)   # (2N, D)
    Z = F.normalize(Z, dim=1)         # unit-normalise
    return Z @ Z.T                    # (2N, 2N)


# ─────────────────────────────────────────────────────────────────────────────
# 6. NT-Xent Loss  (implemented from scratch — no library)
# ─────────────────────────────────────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy Loss (NT-Xent).

    For a batch of N images with 2N views:
        - Positive pair for view i is view i+N  (and vice versa).
        - All other 2N-2 views are negatives.

    Loss for a single view i:
        ℓ(i) = −log [ exp(sim(z_i, z_j) / τ)
                      / Σ_{k≠i} exp(sim(z_i, z_k) / τ) ]
    where j is i's positive partner.

    The total loss averages over all 2N views.
    """

    def __init__(self, tau: float = 0.5):
        super().__init__()
        self.tau = tau

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (N, D)  projected representations of view 1
            z2: (N, D)  projected representations of view 2
        Returns:
            scalar loss
        """
        N      = z1.size(0)
        device = z1.device

        # ── Build (2N × 2N) similarity matrix, scaled by temperature ─────────
        sim = cosine_similarity_matrix(z1, z2) / self.tau   # (2N, 2N)

        # ── Mask out the diagonal (self-similarity is not a valid negative) ───
        diag_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(diag_mask, float("-inf"))

        # ── Positive pair labels ──────────────────────────────────────────────
        #   view i  (i < N)  → positive is i + N
        #   view i  (i >= N) → positive is i - N
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),   # for rows 0 … N-1
            torch.arange(0, N,     device=device),   # for rows N … 2N-1
        ])   # (2N,)

        # ── Cross-entropy over 2N views ───────────────────────────────────────
        loss = F.cross_entropy(sim, labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 7. Similarity-matrix visualisation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def visualise_similarity_heatmap(
    model:    SimCLR,
    loader:   DataLoader,
    device:   torch.device,
    out_path: str,
    title:    str = "Cosine Similarity Matrix",
    n_imgs:   int = 8,
) -> None:
    """
    Pass a small batch (n_imgs images × 2 views) through the model,
    build the (2N × 2N) cosine similarity matrix, and save a heatmap.
    """
    model.eval()
    view1, view2, _ = next(iter(loader))
    view1 = view1[:n_imgs].to(device)
    view2 = view2[:n_imgs].to(device)

    z1, z2 = model(view1, view2)
    sim     = cosine_similarity_matrix(z1, z2).cpu().numpy()   # (2N, 2N)
    N       = n_imgs

    fig, ax = plt.subplots(figsize=(8, 7))
    im      = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Dividing lines between view-1 and view-2 blocks
    ax.axhline(N - 0.5, color="black", linewidth=1.5, linestyle="--")
    ax.axvline(N - 0.5, color="black", linewidth=1.5, linestyle="--")

    # Tick labels
    labels = [f"v1_{i}" for i in range(N)] + [f"v2_{i}" for i in range(N)]
    ax.set_xticks(range(2 * N)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(2 * N)); ax.set_yticklabels(labels, fontsize=7)

    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Per-batch similarity statistics (before training)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_similarity_stats(
    model:      SimCLR,
    loader:     DataLoader,
    device:     torch.device,
    num_batches: int = 5,
):
    """Return average cosine similarity for same-image vs different-image pairs."""
    model.eval()
    same_sims, diff_sims = [], []

    for batch_idx, (view1, view2, _) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        view1, view2 = view1.to(device), view2.to(device)
        N = view1.size(0)

        # Use encoder only (no projection head) — raw 512-dim features
        z1 = F.normalize(model.encoder(view1), dim=1)
        z2 = F.normalize(model.encoder(view2), dim=1)

        sim_mat = z1 @ z2.T   # (N, N)

        # Diagonal = same-image pairs
        same_sims.extend(sim_mat.diag().cpu().tolist())

        # Off-diagonal = different-image pairs
        off_diag = sim_mat[~torch.eye(N, dtype=torch.bool, device=device)]
        diff_sims.extend(off_diag.cpu().tolist())

    avg_same = sum(same_sims) / len(same_sims)
    avg_diff = sum(diff_sims) / len(diff_sims)
    return avg_same, avg_diff


# ─────────────────────────────────────────────────────────────────────────────
# Main — run Checkpoint 2 tests
# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ── DataLoader (unlabeled SSL split for pre-training context) ─────────────
    base_ds     = get_cifar10_subset(
        DATA_ROOT, f"{SPLITS_DIR}/train_labeled_10percent.txt",
        train=True, transform=None, download=True,
    )
    two_view_ds = TwoViewDataset(base_ds, TwoViewTransform(simclr_transform))
    loader      = DataLoader(two_view_ds, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2, pin_memory=True,
                             drop_last=True)

    # ── Instantiate model ─────────────────────────────────────────────────────
    model = SimCLR().to(device)
    print("Model architecture:")
    print(f"  Encoder output dim      : 512")
    print(f"  Projection output dim   : {PROJ_DIM}")
    print(f"  NT-Xent temperature τ   : {TAU}")

    # ── Pair table ────────────────────────────────────────────────────────────
    describe_pairs(N=4)

    # ── Similarity matrix (random / before training) ──────────────────────────
    print("Generating similarity matrix (random encoder — before training) …")
    visualise_similarity_heatmap(
        model, loader, device,
        f"{RESULTS_DIR}/similarity_matrix_before_training.png",
        title="Cosine Similarity Matrix — Before SimCLR Training (Random Encoder)",
    )

    # ── Similarity statistics before training ────────────────────────────────
    avg_same, avg_diff = compute_similarity_stats(model, loader, device)
    print(f"\nAverage cosine similarity BEFORE training")
    print(f"  Same image, two augmented views : {avg_same:.4f}")
    print(f"  Different images                : {avg_diff:.4f}")

    # ── Sanity-check: NT-Xent loss on a random batch ──────────────────────────
    criterion       = NTXentLoss(tau=TAU)
    v1, v2, _       = next(iter(loader))
    v1, v2          = v1.to(device), v2.to(device)
    z1, z2          = model(v1, v2)
    loss_val        = criterion(z1, z2)
    print(f"\nNT-Xent loss (random encoder, sanity check) : {loss_val.item():.4f}")
    expected_random  = -np.log(1.0 / (2 * BATCH_SIZE - 1))
    print(f"Expected loss for uniform random encoder    : ~{expected_random:.4f}")

    print("\n── Answers to Checkpoint 2 questions ────────────────────────────")
    print(
        "Q: Why is the diagonal ignored in the similarity matrix?\n"
        "   sim(z_i, z_i) = 1 always (unit-normalised dot with itself).\n"
        "   Including it would trivially dominate the denominator.\n"
    )
    print(
        "Q: Where are the positive pairs?\n"
        "   For N images the matrix is 2N×2N, split into four N×N blocks.\n"
        "   Positives are at (i, i+N) and (i+N, i), i.e., the anti-diagonal\n"
        "   entries of the off-diagonal blocks.\n"
    )
    print(
        "Q: Why are all other entries treated as negatives?\n"
        "   They originate from different original images, so the model should\n"
        "   push those representations apart in feature space.\n"
    )


if __name__ == "__main__":
    main()

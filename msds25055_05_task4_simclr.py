
import os, sys
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
from msds25055_05_task2_augmentations import TwoViewTransform, simclr_transform

SEED        = 2026
BATCH_SIZE  = 64
TAU         = 0.5
PROJ_DIM    = 128
DATA_ROOT   = "./data"
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Encoder
# ─────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    ResNet-18 modified for CIFAR-10.

    Modifications vs. torchvision default:
      • conv1   : 7×7 stride-2 → 3×3 stride-1, padding 1
      • maxpool : removed (nn.Identity)
      • fc      : removed — exposes 512-dim feature vector h

    Output shape: (B, 512)
    """
    def __init__(self):
        super().__init__()
        bb = torchvision.models.resnet18(weights=None)
        bb.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bb.maxpool = nn.Identity()
        self.features = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2, bb.layer3, bb.layer4,
            bb.avgpool,          # → (B, 512, 1, 1)
        )

    def forward(self, x):
        return self.features(x).flatten(1)   # (B, 512)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Projection Head
# ─────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    Non-linear projection head used ONLY during SimCLR pre-training.
    Architecture: Linear(512→256) → ReLU → Linear(256→128)

    The encoder h (512-dim) — NOT z (128-dim) — is used downstream.
    """
    def __init__(self, in_dim=512, hid_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, h):
        return self.net(h)   # (B, 128)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SimCLR model
# ─────────────────────────────────────────────────────────────────────────────
class SimCLR(nn.Module):
    """Encoder + Projection Head.  Returns projected representations z1, z2."""
    def __init__(self):
        super().__init__()
        self.encoder    = Encoder()
        self.projection = ProjectionHead()

    def forward(self, view1, view2):
        z1 = self.projection(self.encoder(view1))
        z2 = self.projection(self.encoder(view2))
        return z1, z2


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Cosine Similarity Matrix
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Build the full (2N × 2N) cosine similarity matrix.

    Steps
    -----
    1. Z = concat([z1, z2], dim=0)   → (2N, D)
    2. Normalise each row to unit length.
    3. Return Z @ Z.T                 → (2N, 2N)

    • Diagonal        : sim(z_i, z_i) = 1  → masked out in the loss
    • Positive pairs  : (i, i+N) and (i+N, i)  for i in [0, N)
    • Negative pairs  : all other entries
    """
    Z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2N, D)
    return Z @ Z.T                                         # (2N, 2N)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NT-Xent Loss  (from scratch — no library)
# ─────────────────────────────────────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy Loss.

    For each anchor z_i, the loss is:
        ℓ(i,j) = −log[ exp(sim(z_i,z_j)/τ)  /  Σ_{k≠i} exp(sim(z_i,z_k)/τ) ]
    where j is the positive partner of i.

    Total loss = mean over all 2N anchors.
    """
    def __init__(self, tau: float = 0.5):
        super().__init__()
        self.tau = tau

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N      = z1.size(0)
        device = z1.device

        # (2N, 2N) similarity / temperature
        sim = cosine_similarity_matrix(z1, z2) / self.tau

        # Mask self-similarity (diagonal)
        sim.masked_fill_(torch.eye(2*N, dtype=torch.bool, device=device), float("-inf"))

        # Positive-pair labels:
        #   row i        → column i+N   (for i < N)
        #   row i+N      → column i     (for i < N)
        labels = torch.cat([
            torch.arange(N, 2*N, device=device),
            torch.arange(0, N,   device=device),
        ])

        return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Pair-construction description
# ─────────────────────────────────────────────────────────────────────────────
def describe_pairs(N: int = 4):
    print(f"\nPositive / Negative pair table  (batch N={N}, total views=2N={2*N})")
    print(f"  View-1 indices : 0 … {N-1}")
    print(f"  View-2 indices : {N} … {2*N-1}\n")
    print(f"  {'Original':<14} {'View-1 idx':>12} {'View-2 idx':>12} {'Positive?':>12}")
    print("  " + "-" * 52)
    for i in range(N):
        print(f"  {'image '+str(i):<14} {i:>12} {i+N:>12} {'yes':>12}")
    print("\n  All other (i, k) with k≠i and k≠i+N  →  NEGATIVE pairs\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Similarity heatmap
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def visualise_similarity_heatmap(model, loader, device, out_path,
                                  title="Cosine Similarity Matrix", n=8):
    model.eval()
    v1, v2, _ = next(iter(loader))
    v1, v2    = v1[:n].to(device), v2[:n].to(device)
    z1, z2    = model(v1, v2)
    sim       = cosine_similarity_matrix(z1, z2).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axhline(n-0.5, color="k", lw=1.5, ls="--")
    ax.axvline(n-0.5, color="k", lw=1.5, ls="--")
    tl = [f"v1_{i}" for i in range(n)] + [f"v2_{i}" for i in range(n)]
    ax.set_xticks(range(2*n)); ax.set_xticklabels(tl, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(2*n)); ax.set_yticklabels(tl, fontsize=7)
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # DataLoader
    base_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/train_labeled_10percent.txt",
                                 train=True, transform=None, download=True)
    tv_ds   = TwoViewDataset(base_ds, TwoViewTransform(simclr_transform))
    loader  = DataLoader(tv_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    model = SimCLR().to(device)

    # Pair table
    describe_pairs(N=4)

    # Heatmap before training
    print("Generating similarity matrix (random encoder — before training) …")
    visualise_similarity_heatmap(
        model, loader, device,
        f"{RESULTS_DIR}/similarity_matrix_before_training.png",
        title="Cosine Similarity Matrix — Before SimCLR Training (Random Encoder)",
    )

    # NT-Xent sanity check
    crit = NTXentLoss(tau=TAU)
    with torch.no_grad():
        v1, v2, _ = next(iter(loader))
        v1, v2    = v1.to(device), v2.to(device)
        z1, z2    = model(v1, v2)
        loss_val  = crit(z1, z2).item()
    expected = -np.log(1.0 / (2*BATCH_SIZE - 1))
    print(f"\nNT-Xent loss (random encoder) : {loss_val:.4f}")
    print(f"Expected (uniform baseline)   : {expected:.4f}")

    print("\n── Answers ─────────────────────────────────────────────────────")
    print("Q: Why is the diagonal ignored?")
    print("   sim(z_i,z_i)=1 always; it is trivial and must not count as a pair.")
    print("Q: Where are the positive pairs?")
    print("   At (i, i+N) and (i+N, i) — the matching off-diagonal block entries.")
    print("Q: Why are all other entries negatives?")
    print("   They come from different original images, so the model must push them apart.")


if __name__ == "__main__":
    main()

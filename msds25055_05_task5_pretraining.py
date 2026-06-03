
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset, TwoViewDataset
from msds25055_05_task2_augmentations import TwoViewTransform, simclr_transform
from msds25055_05_task4_simclr import (
    SimCLR, NTXentLoss, cosine_similarity_matrix, visualise_similarity_heatmap
)

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 2026
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 3e-4
TAU         = 0.5
DATA_ROOT   = "./data"
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"
GRAPHS_DIR  = "./graphs"
MODELS_DIR  = "./models"

for d in [RESULTS_DIR, GRAPHS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Similarity statistics ─────────────────────────────────────────────────────
@torch.no_grad()
def compute_sim_stats(model, loader, device, num_batches=10):
    """Average cosine similarity: same-image pair vs different-image pair."""
    model.eval()
    same_sims, diff_sims = [], []
    for idx, (v1, v2, _) in enumerate(loader):
        if idx >= num_batches:
            break
        v1, v2 = v1.to(device), v2.to(device)
        N  = v1.size(0)
        z1 = F.normalize(model.encoder(v1), dim=1)
        z2 = F.normalize(model.encoder(v2), dim=1)
        S  = z1 @ z2.T
        same_sims.extend(S.diag().cpu().tolist())
        off = S[~torch.eye(N, dtype=torch.bool, device=device)]
        diff_sims.extend(off.cpu().tolist())
    return sum(same_sims)/len(same_sims), sum(diff_sims)/len(diff_sims)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_simclr(model, loader, optimizer, criterion, device, epochs):
    model.train()
    losses = []
    for ep in range(1, epochs + 1):
        ep_loss, n = 0.0, 0
        for v1, v2, _ in loader:
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(v1, v2)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n       += 1
        avg = ep_loss / n
        losses.append(avg)
        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{epochs}  NT-Xent loss: {avg:.4f}")
    return losses


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Unlabelled SSL split — labels must NOT be used
    base_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/train_ssl_unlabeled.txt",
                                 train=True, transform=None, download=True)
    tv_ds   = TwoViewDataset(base_ds, TwoViewTransform(simclr_transform))
    loader  = DataLoader(tv_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)
    print(f"Unlabelled SSL train size: {len(base_ds):,}")

    model     = SimCLR().to(device)
    criterion = NTXentLoss(tau=TAU)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Similarity before training ────────────────────────────────────────────
    print("\nMeasuring similarity BEFORE training …")
    sim_same_before, sim_diff_before = compute_sim_stats(model, loader, device)
    visualise_similarity_heatmap(
        model, loader, device,
        f"{RESULTS_DIR}/similarity_matrix_before_training.png",
        title="Similarity Matrix — Before SimCLR Training",
    )

    # ── Pre-training ──────────────────────────────────────────────────────────
    print(f"\nStarting SimCLR pre-training  ({EPOCHS} epochs) …\n")
    losses = train_simclr(model, loader, optimizer, criterion, device, EPOCHS)

    # ── Loss curve ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, EPOCHS+1), losses, lw=2, label="NT-Xent Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent Loss")
    ax.set_title("SimCLR Pre-training Loss Curve  (CIFAR-10 unlabelled)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{GRAPHS_DIR}/simclr_pretraining_loss.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved → {GRAPHS_DIR}/simclr_pretraining_loss.png")

    # ── Similarity after training ─────────────────────────────────────────────
    print("\nMeasuring similarity AFTER training …")
    sim_same_after, sim_diff_after = compute_sim_stats(model, loader, device)
    visualise_similarity_heatmap(
        model, loader, device,
        f"{RESULTS_DIR}/similarity_matrix_after_training.png",
        title="Similarity Matrix — After SimCLR Training",
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Feature Similarity Before vs After SimCLR Pre-training")
    print("=" * 60)
    print(f"{'Pair Type':<42} {'Before':>8}  {'After':>8}")
    print("-" * 60)
    print(f"{'Same image, two augmented views':<42} {sim_same_before:>8.4f}  {sim_same_after:>8.4f}")
    print(f"{'Different images':<42} {sim_diff_before:>8.4f}  {sim_diff_after:>8.4f}")
    print("=" * 60)
    print("\nInterpretation:")
    print("  After SimCLR, same-image similarity should be noticeably higher,")
    print("  confirming the encoder has learnt to bring positive pairs together.")

    # ── Save encoder ──────────────────────────────────────────────────────────
    torch.save(model.encoder.state_dict(), f"{MODELS_DIR}/simclr_encoder.pt")
    print(f"\nSaved → {MODELS_DIR}/simclr_encoder.pt")

    return {
        "sim_same_before": sim_same_before, "sim_diff_before": sim_diff_before,
        "sim_same_after":  sim_same_after,  "sim_diff_after":  sim_diff_after,
        "final_loss":      losses[-1],
    }


if __name__ == "__main__":
    stats = main()

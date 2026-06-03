
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset, TwoViewDataset
from rollNumber_05_task2_augmentations import TwoViewTransform, simclr_transform
from rollNumber_05_task4_simclr import Encoder   # reuse the encoder definition

SEED       = 2026
BATCH_SIZE = 64
DATA_ROOT  = "./data"
SPLITS_DIR = "./splits"


@torch.no_grad()
def compute_similarities(encoder, loader, device, num_batches=10):
    """Return avg cosine similarity for same-image vs different-image view pairs."""
    encoder.eval()
    same_sims, diff_sims = [], []

    for idx, (v1, v2, _) in enumerate(loader):
        if idx >= num_batches:
            break
        v1, v2 = v1.to(device), v2.to(device)
        N  = v1.size(0)
        z1 = F.normalize(encoder(v1), dim=1)   # (N, 512)
        z2 = F.normalize(encoder(v2), dim=1)   # (N, 512)
        S  = z1 @ z2.T                          # (N, N)

        # Diagonal = same-image positive pairs
        same_sims.extend(S.diag().cpu().tolist())

        # Off-diagonal = different-image negative pairs
        mask = ~torch.eye(N, dtype=torch.bool, device=device)
        diff_sims.extend(S[mask].cpu().tolist())

    return sum(same_sims) / len(same_sims), sum(diff_sims) / len(diff_sims)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/train_labeled_10percent.txt",
                                 train=True, transform=None, download=True)
    tv_ds   = TwoViewDataset(base_ds, TwoViewTransform(simclr_transform))
    loader  = DataLoader(tv_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    # Randomly initialised encoder — no pretrained weights loaded
    encoder = Encoder().to(device)
    print("Encoder: randomly initialised (no pretraining)\n")

    avg_same, avg_diff = compute_similarities(encoder, loader, device, num_batches=10)

    print("=" * 55)
    print("Feature Similarity  BEFORE  SimCLR Training")
    print("=" * 55)
    print(f"  Same image, two augmented views : {avg_same:+.4f}")
    print(f"  Different images                : {avg_diff:+.4f}")
    print("=" * 55)
    print("\nObservation:")
    print("  Both values are close — the random encoder has no knowledge")
    print("  that two views of the same image should be nearby.")

    return avg_same, avg_diff


if __name__ == "__main__":
    main()

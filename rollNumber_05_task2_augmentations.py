"""
Task 2 — Understanding Augmentations
=====================================
Implements the SimCLR augmentation pipeline, the TwoViewTransform wrapper,
and generates a visualisation showing:
    Original Image | Augmented View 1 | Augmented View 2
for at least 10 images.

Expected output
---------------
results/augmentation_examples.png
"""

import os
import sys

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_datasets, load_split_indices

# ── constants ──────────────────────────────────────────────────────────────────
SEED        = 2026
RESULTS_DIR = "results"
SPLITS_DIR  = "splits"
NUM_EXAMPLES = 10   # rows in the visualisation grid

# ── augmentation pipeline (as specified in the assignment) ─────────────────────
simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std =(0.2470, 0.2435, 0.2616)),
])


class TwoViewTransform:
    """Returns two independently augmented views of the same PIL image."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


# ── utilities ──────────────────────────────────────────────────────────────────

def unnormalise(tensor):
    """Convert a normalised CIFAR-10 tensor back to [0, 1] for display."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def pil_to_tensor(img):
    """Convert a PIL image to a [0,1] float tensor (for the 'Original' column)."""
    return TF.to_tensor(img)


def save_augmentation_grid(examples, save_path, n=NUM_EXAMPLES):
    """
    examples : list of (pil_image, view1_tensor, view2_tensor)
    Produces a grid with columns [Original, View 1, View 2].
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(7, 2.4 * n))
    fig.suptitle("SimCLR Augmentation Examples\n"
                 "Original  |  Augmented View 1  |  Augmented View 2",
                 fontsize=12, y=1.01)

    col_titles = ["Original", "View 1", "View 2"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, (orig_pil, v1, v2) in enumerate(examples[:n]):
        orig_np = np.array(orig_pil) / 255.0
        v1_np   = unnormalise(v1).permute(1, 2, 0).numpy()
        v2_np   = unnormalise(v2).permute(1, 2, 0).numpy()

        for col, img_np in enumerate([orig_np, v1_np, v2_np]):
            axes[row, col].imshow(img_np)
            axes[row, col].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load CIFAR-10 WITHOUT any transform so we keep the raw PIL images
    raw_train, _ = get_cifar10_datasets(data_dir="./data")

    # Use val split indices to grab a small representative set of images
    val_indices = load_split_indices(os.path.join(SPLITS_DIR, "val.txt"))

    two_view = TwoViewTransform(simclr_transform)

    examples = []
    for idx in val_indices[:NUM_EXAMPLES]:
        pil_img, _ = raw_train[idx]        # raw PIL image
        v1, v2     = two_view(pil_img)     # two augmented tensors
        examples.append((pil_img, v1, v2))

    save_augmentation_grid(
        examples,
        save_path=os.path.join(RESULTS_DIR, "augmentation_examples.png"),
    )

    # ── quick sanity check ────────────────────────────────────────────────────
    pil_img, _ = raw_train[val_indices[0]]
    v1, v2     = two_view(pil_img)
    identical  = torch.allclose(v1, v2)
    print(f"\nSanity check — are the two views identical? {identical}")
    print("  (They should NOT be identical; each view is independently augmented.)")

    print("\nTask 2 complete.")
    print("  Answers to report questions:")
    print("  1. Are the two views identical? → No. Each view uses a fresh random augmentation.")
    print("  2. Do they represent the same object? → Yes; both come from the same source image.")
    print("  3. Why treat them as a positive pair? → They share the same semantic content.")
    print("  4. Too-weak augmentations → easy task, trivial representations (no invariance).")
    print("  5. Too-strong augmentations → views may lose their shared identity; loss diverges.")


if __name__ == "__main__":
    main()

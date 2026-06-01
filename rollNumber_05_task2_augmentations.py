"""
=============================================================
Assignment 5 — Checkpoint 1 | Task 2
Augmentation Pipeline + TwoViewTransform + Visualisation
=============================================================
Outputs
-------
  results/augmentation_examples.png
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 2026
DATA_ROOT   = "./data"
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── SimCLR augmentation pipeline (exactly as specified) ──────────────────────
simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

plain_transform = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


# ── Two-View Transform (implemented from scratch — no library) ─────────────────
class TwoViewTransform:
    """
    Applies `transform` TWICE independently to produce two
    differently-augmented views of the same PIL image.

        view1, view2 = TwoViewTransform(simclr_transform)(pil_img)
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


# ── Helper ────────────────────────────────────────────────────────────────────
def denorm(t: torch.Tensor) -> torch.Tensor:
    """Reverse CIFAR-10 normalisation for display."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


# ── Visualisation ─────────────────────────────────────────────────────────────
def visualize_augmentations(n: int = 10):
    set_seed(SEED)
    raw_ds   = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/train_labeled_10percent.txt",
                                  train=True, transform=None, download=True)
    two_view = TwoViewTransform(simclr_transform)

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.5),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.4})
    for col, title in enumerate(["Original", "Augmented View 1", "Augmented View 2"]):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=6)

    for i in range(n):
        img_pil, label = raw_ds[i]
        v1, v2 = two_view(img_pil)
        orig   = denorm(plain_transform(img_pil)).permute(1,2,0).numpy()
        imgs   = [orig,
                  denorm(v1).permute(1,2,0).numpy(),
                  denorm(v2).permute(1,2,0).numpy()]
        for col, img in enumerate(imgs):
            axes[i, col].imshow(img)
            axes[i, col].axis("off")
        axes[i, 0].set_ylabel(CIFAR10_CLASSES[label], fontsize=9,
                               rotation=0, labelpad=44, va="center")

    fig.suptitle("SimCLR Augmentation Examples  (Original | View 1 | View 2)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = f"{RESULTS_DIR}/augmentation_examples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    visualize_augmentations(n=10)
    print("\n── Task 2 conceptual answers ────────────────────────────────────")
    answers = [
        ("Are the two views identical?",
         "No — each is an independent random draw from the pipeline."),
        ("Do they still represent the same object?",
         "Yes — all augmentations are class-preserving (crop, flip, jitter, grayscale)."),
        ("Why treat them as a positive pair?",
         "Both originate from the same image; a good encoder should embed them nearby."),
        ("What if augmentations are too weak?",
         "Views look almost identical → trivial task → model learns little."),
        ("What if augmentations are too strong?",
         "Views may lose shared semantics → positives look like negatives → training breaks."),
    ]
    for q, a in answers:
        print(f"  Q: {q}\n  A: {a}\n")


if __name__ == "__main__":
    main()

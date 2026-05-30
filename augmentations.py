

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 2026
DATA_ROOT   = "./data"
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SimCLR Augmentation Pipeline  (as specified in the assignment)
# ─────────────────────────────────────────────────────────────────────────────
simclr_transform = T.Compose([
    T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# Plain transform — used only to show "original" in the visualisation
plain_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Two-View Transform  (must be implemented by the student, not from a library)
# ─────────────────────────────────────────────────────────────────────────────
class TwoViewTransform:
    """
    Applies the given transform TWICE independently to produce two
    different augmented views of the same image.

    Usage
    -----
    transform = TwoViewTransform(simclr_transform)
    view1, view2 = transform(pil_image)
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        view1 = self.transform(x)   # first random augmentation
        view2 = self.transform(x)   # second independent augmentation
        return view1, view2


# ─────────────────────────────────────────────────────────────────────────────
# Utility: reverse normalisation for display
# ─────────────────────────────────────────────────────────────────────────────
def denorm(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse CIFAR-10 normalisation so pixel values lie in [0, 1]."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def visualize_augmentations(num_examples: int = 10):
    set_seed(SEED)

    # Load raw PIL images (no transform) so we can apply transforms ourselves
    raw_ds = get_cifar10_subset(
        DATA_ROOT, f"{SPLITS_DIR}/train_labeled_10percent.txt",
        train=True, transform=None, download=True,
    )

    two_view = TwoViewTransform(simclr_transform)

    fig, axes = plt.subplots(
        num_examples, 3,
        figsize=(9, num_examples * 2.4),
        gridspec_kw={"wspace": 0.05, "hspace": 0.4},
    )

    col_titles = ["Original", "Augmented View 1", "Augmented View 2"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=8)

    for i in range(num_examples):
        img_pil, label = raw_ds[i]
        class_name     = CIFAR10_CLASSES[label]

        orig          = denorm(plain_transform(img_pil)).permute(1, 2, 0).numpy()
        v1, v2        = two_view(img_pil)
        view1         = denorm(v1).permute(1, 2, 0).numpy()
        view2         = denorm(v2).permute(1, 2, 0).numpy()

        for col, img in enumerate([orig, view1, view2]):
            ax = axes[i, col]
            ax.imshow(img)
            ax.axis("off")

        # Label on the left
        axes[i, 0].set_ylabel(class_name, fontsize=10, rotation=0,
                               labelpad=40, va="center")

    fig.suptitle(
        "SimCLR Augmentation Examples — Original | View 1 | View 2",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = f"{RESULTS_DIR}/augmentation_examples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    visualize_augmentations(num_examples=10)

    print("\n── Answers to Task 2 questions ──────────────────────────────────")
    print(
        "Q1. Are the two augmented views identical?"
        "    No. Each view is produced by an independent random draw from the\n"
        "    pipeline (random crop, flip, colour jitter, grayscale), so they differ.\n"
    )
    print(
        "Q2. Do they still represent the same object?\n"
        "    Yes. Every augmentation used is class-preserving: a cropped/flipped\n"
        "    cat is still a cat.\n"
    )
    print(
        "Q3. Why should SimCLR treat them as a positive pair?\n"
        "    Both views originate from the same image, so a strong encoder should\n"
        "    map them to nearby points in feature space.\n"
    )
    print(
        "Q4. What if augmentations are too weak?\n"
        "    The two views look nearly identical; the contrastive task becomes\n"
        "    trivial and the model learns little beyond pixel matching.\n"
    )
    print(
        "Q5. What if augmentations are too strong?\n"
        "    The views may no longer share semantic content (e.g., a heavily\n"
        "    cropped and grayscaled image loses its distinguishing features), so\n"
        "    positives look like negatives and contrastive training breaks down.\n"
    )


if __name__ == "__main__":
    main()

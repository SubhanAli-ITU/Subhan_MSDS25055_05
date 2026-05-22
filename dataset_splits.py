import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def load_split_indices(filepath):
    """Load image indices from a split .txt file."""
    with open(filepath, "r") as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    return indices


def get_cifar10_datasets(data_dir="./data"):
    """Return raw CIFAR-10 train and test datasets (no transforms applied yet)."""
    # Used internally; transforms are applied per-task
    raw_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
    raw_test  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)
    return raw_train, raw_test


def get_split_dataset(base_dataset, split_file, transform):
    """
    Return a Subset of base_dataset using indices from split_file,
    with the given transform applied via a wrapper.
    """
    indices = load_split_indices(split_file)
    subset  = Subset(base_dataset, indices)
    return TransformSubset(subset, transform)


class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies a transform to each sample."""

    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

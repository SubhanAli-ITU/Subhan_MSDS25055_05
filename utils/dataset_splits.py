"""Load fixed CIFAR-10 split files."""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10

def read_split_indices(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Split file not found: {path}')
    lines = [l.strip() for l in path.read_text(encoding='utf-8').splitlines()]
    return [int(l) for l in lines if l]

def get_cifar10_subset(data_root, split_file, train,
                       transform=None, target_transform=None, download=False):
    dataset = CIFAR10(root=str(data_root), train=train,
                      transform=transform,
                      target_transform=target_transform,
                      download=download)
    indices = read_split_indices(split_file)
    return Subset(dataset, indices)

class TwoViewDataset(Dataset):
    def __init__(self, base_dataset, two_view_transform):
        self.base_dataset = base_dataset
        self.two_view_transform = two_view_transform
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        view1, view2 = self.two_view_transform(image)
        return view1, view2, target

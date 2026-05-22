"""
Task 1 — Supervised Baseline with Limited Labels
================================================
Trains a ResNet-18 from scratch on the fixed 10 % labeled CIFAR-10 split.

Expected outputs
----------------
graphs/supervised_loss.png
results/supervised_confusion_matrix.png
"""

import os
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── project utilities ──────────────────────────────────────────────────────────
import sys
sys.path.append(os.path.dirname(__file__))
from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_datasets, get_split_dataset

# ── constants ──────────────────────────────────────────────────────────────────
SEED        = 2026
BATCH_SIZE  = 64
LR          = 3e-4
EPOCHS      = 30          # more epochs helps with only 10 % labels
NUM_CLASSES = 10

SPLITS_DIR  = "splits"
GRAPHS_DIR  = "graphs"
RESULTS_DIR = "results"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── helpers ────────────────────────────────────────────────────────────────────

def build_model(num_classes=10):
    """ResNet-18 modified for CIFAR-10 (3×3 conv, no maxpool)."""
    model = torchvision.models.resnet18(weights=None)
    # Replace first conv: 7×7 stride-2 → 3×3 stride-1
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove aggressive downsampling designed for 224×224 ImageNet images
    model.maxpool = nn.Identity()
    # Replace classifier head
    model.fc      = nn.Linear(512, num_classes)
    return model


def get_transforms():
    """Standard augmentation for supervised training on CIFAR-10."""
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std =(0.2470, 0.2435, 0.2616)),
    ])
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std =(0.2470, 0.2435, 0.2616)),
    ])
    return train_transform, eval_transform


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def save_loss_plot(train_losses, val_losses, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss", color="#2196F3")
    ax.plot(val_losses,   label="Val Loss",   color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Supervised Baseline — Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def save_confusion_matrix(labels, preds, class_names, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm   = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Supervised Baseline — Test Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(GRAPHS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    train_transform, eval_transform = get_transforms()

    raw_train, raw_test = get_cifar10_datasets(data_dir="./data")

    train_dataset = get_split_dataset(raw_train,
                                      os.path.join(SPLITS_DIR, "train_labeled_10percent.txt"),
                                      train_transform)
    val_dataset   = get_split_dataset(raw_train,
                                      os.path.join(SPLITS_DIR, "val.txt"),
                                      eval_transform)
    test_dataset  = get_split_dataset(raw_test,
                                      os.path.join(SPLITS_DIR, "test.txt"),
                                      eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train size : {len(train_dataset)}")
    print(f"Val size   : {len(val_dataset)}")
    print(f"Test size  : {len(test_dataset)}")

    # ── model, loss, optimiser ────────────────────────────────────────────────
    model     = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        # Save best checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/supervised_best.pt")

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

    # ── plots ─────────────────────────────────────────────────────────────────
    save_loss_plot(train_losses, val_losses,
                   save_path=os.path.join(GRAPHS_DIR, "supervised_loss.png"))

    # ── final test evaluation ─────────────────────────────────────────────────
    model.load_state_dict(torch.load("models/supervised_best.pt", map_location=device))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print(f"\nFinal Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f} %)")
    print(f"Final Test Loss     : {test_loss:.4f}")

    save_confusion_matrix(test_labels, test_preds, CIFAR10_CLASSES,
                          save_path=os.path.join(RESULTS_DIR, "supervised_confusion_matrix.png"))

    # ── persist metrics for later aggregation ─────────────────────────────────
    metrics = {
        "supervised_10percent_test_acc": round(test_acc, 4),
        "supervised_10percent_test_loss": round(test_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
    }
    with open(os.path.join(RESULTS_DIR, "supervised_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics → {RESULTS_DIR}/supervised_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()

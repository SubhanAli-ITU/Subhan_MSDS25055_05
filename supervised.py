

import os
import sys

# ── Make sure utils/ is importable ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset
from utils.metrics import save_confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 2026
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 3e-4
DATA_ROOT   = "./data"
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"
GRAPHS_DIR  = "./graphs"
MODELS_DIR  = "./models"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def get_resnet18_cifar10(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 modified for CIFAR-10:
      • conv1  : 3×3, stride 1, padding 1  (instead of 7×7 stride 2)
      • maxpool: replaced with Identity    (removes aggressive downsampling)
      • fc     : 512 → num_classes
    """
    model = torchvision.models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval helpers
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    for d in [RESULTS_DIR, GRAPHS_DIR, MODELS_DIR]:
        os.makedirs(d, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = get_cifar10_subset(
        DATA_ROOT, f"{SPLITS_DIR}/train_labeled_10percent.txt",
        train=True, transform=train_transform, download=True,
    )
    val_ds = get_cifar10_subset(
        DATA_ROOT, f"{SPLITS_DIR}/val.txt",
        train=True, transform=eval_transform,
    )
    test_ds = get_cifar10_subset(
        DATA_ROOT, f"{SPLITS_DIR}/test.txt",
        train=False, transform=eval_transform,
    )

    print(f"Train (labeled 10%) : {len(train_ds):,} images")
    print(f"Validation          : {len(val_ds):,} images")
    print(f"Test                : {len(test_ds):,} images")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model / Loss / Optimizer ──────────────────────────────────────────────
    model     = get_resnet18_cifar10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_acc, best_weights = 0.0, None

    print(f"\nTraining for {EPOCHS} epochs …\n")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_acc > best_val_acc:
            best_val_acc  = vl_acc
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  |  "
                  f"Train loss={tr_loss:.4f} acc={tr_acc:.3f}  |  "
                  f"Val loss={vl_loss:.4f} acc={vl_acc:.3f}")

    # ── Save loss curve ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", linewidth=2)
    ax.plot(range(1, EPOCHS + 1), val_losses,   label="Val Loss",   linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Supervised Baseline — Training & Validation Loss\n"
                 "(ResNet-18, 10% labeled CIFAR-10)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_loss = f"{GRAPHS_DIR}/supervised_loss.png"
    fig.savefig(path_loss, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {path_loss}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    model.load_state_dict(best_weights)
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy (best val checkpoint) : {test_acc:.4f}")

    path_cm = f"{RESULTS_DIR}/supervised_confusion_matrix.png"
    save_confusion_matrix(test_labels, test_preds, path_cm,
                          title="Supervised Baseline — Confusion Matrix")
    print(f"Saved → {path_cm}")

    # ── Save model weights ────────────────────────────────────────────────────
    path_model = f"{MODELS_DIR}/supervised_model.pt"
    torch.save(model.state_dict(), path_model)
    print(f"Saved → {path_model}")

    return test_acc


if __name__ == "__main__":
    acc = main()
    print(f"\n[Task 1] supervised_10percent_test_acc = {acc:.4f}")

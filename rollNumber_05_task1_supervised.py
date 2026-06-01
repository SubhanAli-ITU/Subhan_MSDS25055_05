"""
=============================================================
Assignment 5 — Checkpoint 1 | Task 1
Supervised Baseline — ResNet-18 on 10% labeled CIFAR-10
=============================================================
Outputs
-------
  graphs/supervised_loss.png
  results/supervised_confusion_matrix.png
  models/supervised_model.pt
"""

import os, sys
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

# ── Config ────────────────────────────────────────────────────────────────────
SEED, BATCH, EPOCHS, LR = 2026, 64, 30, 3e-4
DATA_ROOT   = "./data"
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"
GRAPHS_DIR  = "./graphs"
MODELS_DIR  = "./models"
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

for d in [RESULTS_DIR, GRAPHS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────────
def build_resnet18(num_classes=10):
    """ResNet-18 modified for CIFAR-10 (3×3 conv, no maxpool)."""
    m = torchvision.models.resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, num_classes)
    return m

# ── Train / eval helpers ───────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, opt, device):
    model.train()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward(); opt.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    preds_all, labels_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        p  = out.argmax(1)
        correct += (p == y).sum().item()
        total   += x.size(0)
        preds_all.extend(p.cpu().tolist())
        labels_all.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, preds_all, labels_all

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(MEAN, STD)])
    ev_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    train_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/train_labeled_10percent.txt",
                                  train=True,  transform=tr_tf, download=True)
    val_ds   = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/val.txt",
                                  train=True,  transform=ev_tf)
    test_ds  = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/test.txt",
                                  train=False, transform=ev_tf)

    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    kw = dict(batch_size=BATCH, num_workers=2, pin_memory=True)
    tr_ldr = DataLoader(train_ds, shuffle=True,  **kw)
    vl_ldr = DataLoader(val_ds,   shuffle=False, **kw)
    te_ldr = DataLoader(test_ds,  shuffle=False, **kw)

    model = build_resnet18().to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    tr_losses, vl_losses = [], []
    best_acc, best_sd = 0.0, None

    for ep in range(1, EPOCHS + 1):
        tr_l, tr_a          = train_epoch(model, tr_ldr, crit, opt, device)
        vl_l, vl_a, _, _    = eval_epoch(model, vl_ldr, crit, device)
        tr_losses.append(tr_l); vl_losses.append(vl_l)
        if vl_a > best_acc:
            best_acc = vl_a
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"Ep {ep:3d}/{EPOCHS}  tr_loss={tr_l:.4f} tr_acc={tr_a:.3f}"
                  f"  vl_loss={vl_l:.4f} vl_acc={vl_a:.3f}")

    # ── Loss plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, EPOCHS+1), tr_losses, label="Train Loss", lw=2)
    ax.plot(range(1, EPOCHS+1), vl_losses, label="Val Loss",   lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Supervised Baseline — Loss (ResNet-18, 10% labels)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{GRAPHS_DIR}/supervised_loss.png", dpi=150)
    plt.close(fig); print(f"Saved → {GRAPHS_DIR}/supervised_loss.png")

    # ── Test evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(best_sd)
    _, te_acc, te_preds, te_labels = eval_epoch(model, te_ldr, crit, device)
    print(f"\nTest Accuracy: {te_acc:.4f}")

    save_confusion_matrix(te_labels, te_preds,
                          f"{RESULTS_DIR}/supervised_confusion_matrix.png",
                          title="Supervised Baseline — Confusion Matrix")
    torch.save(model.state_dict(), f"{MODELS_DIR}/supervised_model.pt")
    return te_acc


if __name__ == "__main__":
    acc = main()
    print(f"\n[Task 1] supervised_10percent_test_acc = {acc:.4f}")

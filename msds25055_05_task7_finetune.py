

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset
from msds25055_05_task4_simclr import Encoder

SEED       = 2026
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 3e-4
DATA_ROOT  = "./data"
SPLITS_DIR = "./splits"
GRAPHS_DIR = "./graphs"
MODELS_DIR = "./models"
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

for d in [GRAPHS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Full classification model (encoder + head) ────────────────────────────────
class FineTuneModel(nn.Module):
    def __init__(self, encoder: Encoder, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


# ── Train / eval helpers ───────────────────────────────────────────────────────
def train_epoch(model, loader, crit, opt, device):
    model.train()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out  = model(x)
        loss = crit(out, y)
        loss.backward(); opt.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        loss = crit(out, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(MEAN, STD)])
    ev_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    kw = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    tr_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/train_labeled_10percent.txt",
                               train=True, transform=tr_tf, download=True)
    vl_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/val.txt",  train=True,  transform=ev_tf)
    te_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/test.txt", train=False, transform=ev_tf)

    tr_ldr = DataLoader(tr_ds, shuffle=True,  **kw)
    vl_ldr = DataLoader(vl_ds, shuffle=False, **kw)
    te_ldr = DataLoader(te_ds, shuffle=False, **kw)

    enc_path = f"{MODELS_DIR}/simclr_encoder.pt"
    if not os.path.exists(enc_path):
        print(f"SimCLR encoder not found at {enc_path}.")
        print("Run rollNumber_05_task5_pretraining.py first.")
        return None

    encoder = Encoder()
    encoder.load_state_dict(torch.load(enc_path, map_location="cpu"))
    model = FineTuneModel(encoder).to(device)
    print(f"Loaded SimCLR encoder from {enc_path}")
    print(f"Fine-tuning FULL model for {EPOCHS} epochs …\n")

    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=LR)

    tr_accs, vl_accs = [], []
    best_acc, best_sd = 0.0, None

    for ep in range(1, EPOCHS + 1):
        tr_l, tr_a = train_epoch(model, tr_ldr, crit, opt, device)
        vl_l, vl_a = eval_epoch(model, vl_ldr, crit, device)
        tr_accs.append(tr_a); vl_accs.append(vl_a)
        if vl_a > best_acc:
            best_acc = vl_a
            best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:3d}/{EPOCHS}  tr_loss={tr_l:.4f} tr_acc={tr_a:.3f}"
                  f"  vl_loss={vl_l:.4f} vl_acc={vl_a:.3f}")

    model.load_state_dict(best_sd)
    _, te_acc = eval_epoch(model, te_ldr, crit, device)
    print(f"\nFine-tune Test Accuracy: {te_acc:.4f}")

    # ── Accuracy curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ep_r = range(1, EPOCHS+1)
    ax.plot(ep_r, tr_accs, lw=2, label="Train Acc")
    ax.plot(ep_r, vl_accs, lw=2, label="Val Acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title(f"SimCLR Fine-tuning Accuracy  (test={te_acc:.3f})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{GRAPHS_DIR}/finetuning_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {GRAPHS_DIR}/finetuning_accuracy.png")

    torch.save(model.state_dict(), f"{MODELS_DIR}/finetuned_model.pt")
    print(f"Saved → {MODELS_DIR}/finetuned_model.pt")

    return te_acc


if __name__ == "__main__":
    acc = main()
    if acc:
        print(f"\n[Task 7] simclr_finetune_test_acc = {acc:.4f}")

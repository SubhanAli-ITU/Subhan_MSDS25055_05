"""
=============================================================
Assignment 5 — Checkpoint 4 | Task 6
Linear Probe Evaluation
=============================================================
Experiment A : Random  frozen encoder + linear head
Experiment B : SimCLR  frozen encoder + linear head

Outputs
-------
  graphs/linear_probe_accuracy.png
  models/linear_probe.pt
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset
from rollNumber_05_task4_simclr import Encoder

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


# ── Extract features from a frozen encoder ────────────────────────────────────
@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(encoder(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


# ── Train a linear head on pre-extracted features ─────────────────────────────
def train_linear_head(linear, tr_X, tr_y, vl_X, vl_y, device, epochs, lr):
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(linear.parameters(), lr=lr)
    tr_X, tr_y = tr_X.to(device), tr_y.to(device)
    vl_X, vl_y = vl_X.to(device), vl_y.to(device)

    tr_accs, vl_accs = [], []
    for ep in range(1, epochs + 1):
        linear.train()
        opt.zero_grad()
        out  = linear(tr_X)
        loss = crit(out, tr_y)
        loss.backward(); opt.step()
        tr_acc = (out.argmax(1) == tr_y).float().mean().item()

        linear.eval()
        with torch.no_grad():
            vl_acc = (linear(vl_X).argmax(1) == vl_y).float().mean().item()

        tr_accs.append(tr_acc); vl_accs.append(vl_acc)
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:3d}/{epochs}  tr_acc={tr_acc:.3f}  vl_acc={vl_acc:.3f}")
    return tr_accs, vl_accs


@torch.no_grad()
def test_linear(linear, te_X, te_y, device):
    linear.eval()
    te_X, te_y = te_X.to(device), te_y.to(device)
    return (linear(te_X).argmax(1) == te_y).float().mean().item()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    kw = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)

    tr_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/train_labeled_10percent.txt",
                               train=True, transform=tf, download=True)
    vl_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/val.txt",  train=True,  transform=tf)
    te_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/test.txt", train=False, transform=tf)

    tr_ldr = DataLoader(tr_ds, **kw)
    vl_ldr = DataLoader(vl_ds, **kw)
    te_ldr = DataLoader(te_ds, **kw)

    results = {}

    for label, enc_path in [("Random",  None),
                             ("SimCLR",  f"{MODELS_DIR}/simclr_encoder.pt")]:
        print(f"\n{'='*55}")
        print(f"Experiment: {label} frozen encoder + linear head")
        print("="*55)

        encoder = Encoder().to(device)
        if enc_path:
            if not os.path.exists(enc_path):
                print(f"  SimCLR encoder not found at {enc_path}.")
                print("  Run rollNumber_05_task5_pretraining.py first.")
                continue
            encoder.load_state_dict(torch.load(enc_path, map_location=device))
            print(f"  Loaded SimCLR encoder from {enc_path}")
        else:
            print("  Using randomly initialised encoder (no weights loaded)")

        print("  Extracting features …")
        tr_X, tr_y = extract_features(encoder, tr_ldr, device)
        vl_X, vl_y = extract_features(encoder, vl_ldr, device)
        te_X, te_y = extract_features(encoder, te_ldr, device)

        linear = nn.Linear(512, 10).to(device)
        tr_accs, vl_accs = train_linear_head(
            linear, tr_X, tr_y, vl_X, vl_y, device, EPOCHS, LR)
        te_acc = test_linear(linear, te_X, te_y, device)
        print(f"\n  [{label}] Test accuracy: {te_acc:.4f}")
        results[label] = {"tr": tr_accs, "vl": vl_accs, "te": te_acc, "linear": linear}

    # ── Accuracy curves ───────────────────────────────────────────────────────
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (label, res) in zip(axes, results.items()):
        ep_range = range(1, EPOCHS+1)
        ax.plot(ep_range, res["tr"], label="Train Acc", lw=2)
        ax.plot(ep_range, res["vl"], label="Val Acc",   lw=2)
        ax.set_title(f"{label} Encoder  (test={res['te']:.3f})", fontsize=12)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Linear Probe Accuracy — Random vs SimCLR Encoder", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{GRAPHS_DIR}/linear_probe_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved → {GRAPHS_DIR}/linear_probe_accuracy.png")

    # Save SimCLR linear probe weights
    if "SimCLR" in results:
        torch.save(results["SimCLR"]["linear"].state_dict(), f"{MODELS_DIR}/linear_probe.pt")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Model':<40} {'Test Acc':>10}")
    print("-" * 55)
    for label, res in results.items():
        print(f"  {label+' encoder linear probe':<38} {res['te']:>10.4f}")
    print("=" * 55)

    return {k: v["te"] for k, v in results.items()}


if __name__ == "__main__":
    accs = main()

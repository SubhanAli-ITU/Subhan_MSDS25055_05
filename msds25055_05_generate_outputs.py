

import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_subset
from msds25055_05_task4_simclr import Encoder
from msds25055_05_task7_finetune import FineTuneModel

SEED        = 2026
BATCH_SIZE  = 64
DATA_ROOT   = "./data"
SPLITS_DIR  = "./splits"
RESULTS_DIR = "./results"
MODELS_DIR  = "./models"
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Load test set ──────────────────────────────────────────────────────────────
def get_test_loader():
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/test.txt",
                            train=False, transform=tf, download=True)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ── Evaluate any model on test set ────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(1)
        correct += (preds == y).sum().item()
        total   += x.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return correct / total, all_preds, all_labels, all_probs


def try_load(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return True
    print(f"  WARNING: {path} not found — skipping")
    return False


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_loader = get_test_loader()

    # ── 1. Supervised baseline accuracy ───────────────────────────────────────
    from msds25055_05_task1_supervised import build_resnet18
    sup_model = build_resnet18(10).to(device)
    sup_acc = 0.0
    if try_load(sup_model, f"{MODELS_DIR}/supervised_model.pt", device):
        sup_acc, *_ = evaluate_model(sup_model, test_loader, device)
    print(f"[1] Supervised 10% test acc   : {sup_acc:.4f}")

    # ── 2. Random encoder + linear probe ─────────────────────────────────────
    from msds25055_05_task6_linear_probe import extract_features
    import torchvision.transforms as T
    from utils.dataset_splits import get_cifar10_subset

    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    te_ds = get_cifar10_subset(DATA_ROOT, f"{SPLITS_DIR}/test.txt", train=False, transform=tf)
    te_ldr = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    rand_enc = Encoder().to(device)
    rand_lin = nn.Linear(512, 10).to(device)
    rand_probe_acc = 0.0
    if try_load(rand_lin, f"{MODELS_DIR}/random_linear_probe.pt", device):
        te_X, te_y = extract_features(rand_enc, te_ldr, device)
        te_X, te_y = te_X.to(device), te_y.to(device)
        rand_lin.eval()
        with torch.no_grad():
            rand_probe_acc = (rand_lin(te_X).argmax(1) == te_y).float().mean().item()
    print(f"[2] Random linear probe acc   : {rand_probe_acc:.4f}")

    # ── 3. SimCLR encoder + linear probe ─────────────────────────────────────
    sim_enc = Encoder().to(device)
    sim_lin = nn.Linear(512, 10).to(device)
    sim_probe_acc = 0.0
    enc_ok  = try_load(sim_enc, f"{MODELS_DIR}/simclr_encoder.pt", device)
    lin_ok  = try_load(sim_lin, f"{MODELS_DIR}/linear_probe.pt",   device)
    if enc_ok and lin_ok:
        te_X, te_y = extract_features(sim_enc, te_ldr, device)
        te_X, te_y = te_X.to(device), te_y.to(device)
        sim_lin.eval()
        with torch.no_grad():
            sim_probe_acc = (sim_lin(te_X).argmax(1) == te_y).float().mean().item()
    print(f"[3] SimCLR linear probe acc   : {sim_probe_acc:.4f}")

    # ── 4. SimCLR fine-tuned model ────────────────────────────────────────────
    ft_model = FineTuneModel(Encoder()).to(device)
    ft_acc   = 0.0
    ft_preds = ft_labels = ft_probs = None
    if try_load(ft_model, f"{MODELS_DIR}/finetuned_model.pt", device):
        ft_acc, ft_preds, ft_labels, ft_probs = evaluate_model(ft_model, test_loader, device)
    print(f"[4] SimCLR fine-tune acc      : {ft_acc:.4f}")

    # ── Similarity stats (load from file if saved) ────────────────────────────
    sim_stats_path = f"{RESULTS_DIR}/sim_stats.json"
    sim_same_before = sim_diff_before = sim_same_after = sim_diff_after = 0.0
    if os.path.exists(sim_stats_path):
        with open(sim_stats_path) as f:
            stats = json.load(f)
        sim_same_before = stats.get("sim_same_before", 0.0)
        sim_diff_before = stats.get("sim_diff_before", 0.0)
        sim_same_after  = stats.get("sim_same_after",  0.0)
        sim_diff_after  = stats.get("sim_diff_after",  0.0)

    # ── Write metrics.json ────────────────────────────────────────────────────
    metrics = {
        "student_name":                "YourName",
        "roll_number":                 "YourRollNumber",
        "seed":                        SEED,
        "batch_size":                  BATCH_SIZE,
        "simclr_epochs":               50,
        "linear_probe_epochs":         20,
        "finetuning_epochs":           20,
        "learning_rate":               LR if "LR" in dir() else 3e-4,
        "temperature":                 0.5,
        "supervised_10percent_test_acc": round(sup_acc,        4),
        "random_linear_probe_test_acc":  round(rand_probe_acc, 4),
        "simclr_linear_probe_test_acc":  round(sim_probe_acc,  4),
        "simclr_finetune_test_acc":      round(ft_acc,         4),
        "same_view_similarity_before":   round(sim_same_before, 4),
        "different_image_similarity_before": round(sim_diff_before, 4),
        "same_view_similarity_after":    round(sim_same_after,  4),
        "different_image_similarity_after":  round(sim_diff_after,  4),
    }
    metrics_path = f"{RESULTS_DIR}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved → {metrics_path}")

    # ── Write test_predictions.csv ────────────────────────────────────────────
    csv_path = f"{RESULTS_DIR}/test_predictions.csv"
    if ft_preds is not None:
        header = ["image_index", "true_label", "predicted_label"] + \
                 [f"prob_class_{i}" for i in range(10)]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx, (pred, true, probs) in enumerate(zip(ft_preds, ft_labels, ft_probs)):
                writer.writerow([idx, true, pred] + [round(p, 6) for p in probs])
        print(f"Saved → {csv_path}")
    else:
        print(f"Skipping {csv_path} — fine-tuned model not available")

    # ── Final comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("Final Results Summary")
    print("=" * 75)
    print(f"{'Model':<50} {'Labels?':>8} {'Frozen?':>8} {'Test Acc':>10}")
    print("-" * 75)
    rows = [
        ("Supervised ResNet-18, 10% labels",         "Yes", "No",  sup_acc),
        ("Random encoder + linear probe",            "No",  "Yes", rand_probe_acc),
        ("SimCLR encoder + linear probe",            "No",  "Yes", sim_probe_acc),
        ("SimCLR encoder + full fine-tuning",        "Mix", "No",  ft_acc),
    ]
    for name, lbl, frz, acc in rows:
        print(f"  {name:<48} {lbl:>8} {frz:>8} {acc:>10.4f}")
    print("=" * 75)


if __name__ == "__main__":
    LR = 3e-4
    main()

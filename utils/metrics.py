"""Basic metric helpers."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=list(range(10)))

def save_confusion_matrix(y_true, y_pred, out_path, title='Confusion Matrix'):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = compute_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_CLASSES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved → {out_path}')

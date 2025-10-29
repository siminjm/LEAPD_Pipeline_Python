
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(labels, scores, title="ROC Curve", save_path=None):
    labels = np.asarray(labels).astype(int).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_channel_heatmap(channels, values, title="Channel Metric Heatmap", save_path=None):
    r"""
    Simple 2D heatmap-like bar visualization: channels vs value.
    If you later provide 2D scalp coordinates, we can switch to a real topomap.
    r"""
    import matplotlib.pyplot as plt
    import numpy as np
    channels = list(channels)
    values = np.asarray(values).astype(float).reshape(-1)
    order = np.argsort(values)[::-1]
    ch_sorted = [channels[i] for i in order]
    val_sorted = values[order]
    plt.figure()
    plt.bar(np.arange(len(val_sorted)), val_sorted)
    plt.xticks(np.arange(len(val_sorted)), ch_sorted, rotation=90)
    plt.ylabel("Metric")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

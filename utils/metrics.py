from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

def evaluate_classification(scores: np.ndarray, labels: np.ndarray):
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)
    pred = (scores >= 0.5).astype(int)
    ACC = (pred == labels).mean() * 100.0
    TP = ((pred==1) & (labels==1)).sum()
    TN = ((pred==0) & (labels==0)).sum()
    FP = ((pred==1) & (labels==0)).sum()
    FN = ((pred==0) & (labels==1)).sum()
    SEN = TP / max(TP+FN, 1)
    SPC = TN / max(TN+FP, 1)
    PPV = TP / max(TP+FP, 1)
    NPV = TN / max(TN+FN, 1)
    OR  = (TP*TN) / max(FP*FN, 1)
    LRp = SEN / max(1 - SPC, np.finfo(float).eps)
    try:
        AUC = float(roc_auc_score(labels, scores))
    except Exception:
        AUC = float('nan')
    return {
        'ACC': ACC, 'AUC': AUC, 'SEN': SEN, 'SPC': SPC, 'PPV': PPV, 'NPV': NPV, 'OR': OR, 'LRp': LRp,
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN)
    }

def evaluate_correlation(scores: np.ndarray, target: np.ndarray):
    scores = np.asarray(scores).reshape(-1)
    target = np.asarray(target).reshape(-1)
    mask = np.isfinite(scores) & np.isfinite(target)
    if mask.sum() < 3:
        return {'Rho': np.nan, 'PValue': np.nan, 'N': int(mask.sum())}
    rho, p = spearmanr(scores[mask], target[mask])
    return {'Rho': float(rho), 'PValue': float(p), 'N': int(mask.sum())}

def pick_polarity_and_rho(scores: np.ndarray, y: np.ndarray):
    mask = np.isfinite(scores) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, +1, scores
    rho_keep, p_keep = spearmanr(scores[mask], y[mask])
    sf = 1 - scores
    rho_flip, p_flip = spearmanr(sf[mask], y[mask])
    if abs(rho_flip) > abs(rho_keep):
        return float(rho_flip), float(p_flip), -1, sf
    else:
        return float(rho_keep), float(p_keep), +1, scores

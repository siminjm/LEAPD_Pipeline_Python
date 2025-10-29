from __future__ import annotations
import numpy as np

def build_hyperplanes(V: np.ndarray, d: int):
    """Return basis P (Kxd) and mean m (1xK) using SVD on centered data."""
    V = np.asarray(V, dtype=float)
    m = V.mean(axis=0, keepdims=True)
    Vc = V - m
    # scale like MATLAB: Vc / sqrt(n-1)
    n = Vc.shape[0]
    if n <= 1:
        # degenerate: return zeros
        K = Vc.shape[1]
        return np.zeros((K, min(d, K))), m.squeeze()
    U, S, VT = np.linalg.svd(Vc / np.sqrt(max(n-1, 1)), full_matrices=False)
    Vr = VT.T
    d = min(d, Vr.shape[1])
    P = Vr[:, :d]
    return P, m.squeeze()

def compute_leapd_scores(V: np.ndarray, P0, m0, P1, m1, dstar=None, is_norm: int=0):
    """Distance-to-hyperplane ratio in [0,1]."""
    V = np.atleast_2d(V).astype(float)
    if dstar is None:
        dstar = P0.shape[1]
    Y0 = V - m0
    Y1 = V - m1
    proj0 = (Y0 @ P0) @ P0.T
    proj1 = (Y1 @ P1) @ P1.T
    if is_norm == 0:
        d1 = np.linalg.norm(Y0 - proj0, axis=1)
        d2 = np.linalg.norm(Y1 - proj1, axis=1)
    else:
        d1 = np.linalg.norm(proj0, axis=1) / np.maximum(np.linalg.norm(Y0, axis=1), np.finfo(float).eps)
        d2 = np.linalg.norm(proj1, axis=1) / np.maximum(np.linalg.norm(Y1, axis=1), np.finfo(float).eps)
    s = d2 / np.maximum(d1 + d2, np.finfo(float).eps)
    return s

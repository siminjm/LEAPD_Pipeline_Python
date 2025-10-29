from __future__ import annotations
import numpy as np
from itertools import combinations

def combine_scores(S: np.ndarray) -> np.ndarray:
    """Geometric mean of odds across columns -> score in [0,1]."""
    S = np.asarray(S, dtype=float)
    odds = S / np.maximum(1 - S, np.finfo(float).eps)
    combo_odds = np.exp(np.nanmean(np.log(np.maximum(odds, np.finfo(float).eps)), axis=1))
    combo_score = combo_odds / (1 + combo_odds)
    return combo_score

def generate_combinations(best_prev: dict, Nch: int, k: int):
    """Progressive fixed-best using seeds from <=5-ch best results (indices are 0-based)."""
    if not best_prev:
        return np.array(list(combinations(range(Nch), k)), dtype=int)
    seed = set()
    for t in sorted(best_prev.keys()):
        idx = best_prev[t].get('indices', [])
        seed.update(idx)
    seed = sorted(seed)
    remaining = [i for i in range(Nch) if i not in seed]
    need = k - len(seed)
    if need <= 0:
        return np.array([seed[:k]], dtype=int)
    if not remaining:
        return np.array([seed[:min(len(seed), k)]], dtype=int)
    rem_combs = list(combinations(remaining, need))
    combos = [list(seed) + list(rc) for rc in rem_combs]
    return np.array(combos, dtype=int)

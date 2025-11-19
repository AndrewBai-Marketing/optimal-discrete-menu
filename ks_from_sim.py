from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers shared by train/evaluate
# -----------------------------
def _load_theta(csv_path: str | Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    attr_cols = [c for c in df.columns if c.startswith("wtp_")]
    if not attr_cols:
        raise ValueError("No 'wtp_*' columns found.")
    Theta = df[attr_cols].to_numpy(dtype=float)
    return Theta, attr_cols


# at top
COST_BASE, COST_C1, COST_C2, COST_EXP = 0.0, 5.0, 1.0, 2.0

def _cost_linear(x: np.ndarray, cost_per_attr: float = 5.0, base_cost: float = 0.0) -> float:
    # keep function name for compatibility; implement convex cost
    k = int(np.sum(x))
    return float(COST_BASE + COST_C1*k + (COST_C2 * (k ** COST_EXP)))



def _build_library(d: int) -> list[np.ndarray]:
    lib: list[np.ndarray] = []
    for bits in range(1, 1 << d):
        x = np.array([(bits >> j) & 1 for j in range(d)], dtype=int)
        lib.append(x)
    return lib


def _single_option_price_multi(Theta: np.ndarray, x: np.ndarray, cost: float,
                               p_steps: int = 64, p_quantile: float = 0.995) -> float:
    s = Theta @ x
    s_max = float(np.quantile(s, p_quantile))
    lo = max(cost + 1e-8, 0.0)
    hi = max(lo + 1e-6, s_max)
    if hi <= lo + 1e-9:
        return lo
    grid = np.linspace(lo, hi, p_steps)
    best_p, best_val = lo, -1e18
    M = float(len(s))
    for p in grid:
        tail = float(np.sum(s >= p)) / M
        val = (p - cost) * tail
        if val > best_val:
            best_val, best_p = val, p
    return best_p


# -----------------------------
# Public API expected by runner
# -----------------------------
def train(csv_train: str | Path, L_max: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy GK/KS-style selection with fixed per-item prices (solo prices).

    Returns (bundle_indices, prices) picked on the TRAIN dataset.
    """
    Theta, attr_cols = _load_theta(csv_train)
    M, d = Theta.shape
    lib = _build_library(d)
    K = len(lib)

    # Costs and per-item fixed prices via single-option rule
    C = np.array([_cost_linear(x) for x in lib], dtype=float)
    prices_fixed = np.array([
        max(_single_option_price_multi(Theta, x, C[j]), C[j]) for j, x in enumerate(lib)
    ], dtype=float)
    margins = prices_fixed - C

    # Precompute utilities for all candidates
    Xmat = np.vstack(lib).astype(int)  # (K,d)
    U = Theta @ Xmat.T - prices_fixed[None, :]  # (M,K)

    # Greedy add-by-add up to L_max; track best per L
    chosen: list[int] = []
    best_util = np.full(M, -np.inf, dtype=float)
    best_margin = np.zeros(M, dtype=float)
    best_pp = -1e18
    best_L = 0
    best_set: list[int] = []

    for L in range(1, L_max + 1):
        remaining = [j for j in range(K) if j not in chosen]
        if not remaining:
            break
        # Evaluate marginal gain for each remaining j
        gains = np.zeros(len(remaining), dtype=float)
        for t, j in enumerate(remaining):
            u_j = U[:, j]
            take = u_j > best_util
            # Only count consumers with positive utility under j
            take_pos = take & (u_j > 0)
            # New profit contribution for those consumers if j added
            delta = np.where(best_util > 0, margins[j] - best_margin, margins[j])
            gains[t] = float(np.sum(delta[take_pos])) / M
        # Pick best
        j_star = remaining[int(np.argmax(gains))]
        chosen.append(j_star)
        # Update trackers
        u_j = U[:, j_star]
        better = u_j > best_util
        # For better ones, update both util and margin
        best_margin[better] = margins[j_star]
        best_util[better] = u_j[better]

        # Compute current pp (per-person profit)
        pp_now = float(np.sum(best_margin[best_util > 0])) / M
        if pp_now > best_pp + 1e-12:
            best_pp = pp_now
            best_L = L
            best_set = chosen.copy()

    idx = np.array(best_set, dtype=int)
    return idx, prices_fixed[idx]


def evaluate(csv_test: str | Path, bundles: np.ndarray, prices: np.ndarray) -> Tuple[float, float]:
    """Evaluate fixed-price menu on TEST dataset; returns (pp_ic, outside_share)."""
    Theta, attr_cols = _load_theta(csv_test)
    M, d = Theta.shape
    lib = _build_library(d)
    X = np.vstack([lib[int(j)] for j in bundles]).astype(int)  # (L,d)
    C = np.array([_cost_linear(x) for x in X], dtype=float)
    margins = prices - C
    U = Theta @ X.T - prices.reshape(1, -1)  # (M,L)
    best_j = np.argmax(np.concatenate([U, np.zeros((M, 1))], axis=1), axis=1)
    L = X.shape[0]
    chosen = best_j < L
    picks = best_j[chosen]
    pp = float(np.sum(margins[picks][U[chosen, picks] > 0])) / M
    outside = float(1.0 - np.sum(U.max(axis=1) > 0) / M)
    return pp, outside


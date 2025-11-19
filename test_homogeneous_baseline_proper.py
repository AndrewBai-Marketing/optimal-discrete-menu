"""
Proper train/test evaluation for Homogeneous Baseline scenario
Train on 80%, test on held-out 20%
"""
import ctypes
import numpy as np
import pandas as pd
import time
from itertools import combinations
import platform
import ks_from_sim as ks
from utils_train_test_split import create_train_test_split
from baselines import uniform_pricing, good_better_best, personalization_oracle

# Load DLL (cross-platform)
if platform.system() == 'Windows':
    dll = ctypes.CDLL("./ot_core.dll")
elif platform.system() == 'Darwin':  # macOS
    dll = ctypes.CDLL("./ot_core.dylib")
else:  # Linux
    dll = ctypes.CDLL("./ot_core.so")

def setup_and_run_ot(csv_path, L, restarts=30):
    """Run OT with C++ acceleration on given dataset"""
    df = pd.read_csv(csv_path)
    wtp_cols = [c for c in df.columns if c.startswith('wtp_')]
    wtps = df[wtp_cols].values
    M, d = wtps.shape

    K = 2 ** d - 1
    bundles_list = []
    for size in range(1, d + 1):
        for combo in combinations(range(d), size):
            bundle = np.zeros(d, dtype=int)
            bundle[list(combo)] = 1
            bundles_list.append(bundle)

    bundles_matrix = np.array(bundles_list)
    S = wtps @ bundles_matrix.T
    c1, c2, b, F = 5.0, 1.0, 2.0, 0.0
    bundle_sizes = bundles_matrix.sum(axis=1)
    Cx = F + c1 * bundle_sizes + c2 * (bundle_sizes ** b)

    init_env = dll.init_environment
    init_env.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
    ]
    init_env.restype = None

    S_flat = S.flatten(order='C')
    init_env(
        S_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        M, K, F, c1, c2, b
    )

    lloyd_c = dll.lloyd_menu_c
    lloyd_c.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    ]
    lloyd_c.restype = None

    out_bundles = np.zeros(L, dtype=np.int32)
    out_prices = np.zeros(L, dtype=np.float64)
    out_shares = np.zeros(L, dtype=np.float64)
    out_pp_ic = np.zeros(1, dtype=np.float64)
    out_outside = np.zeros(1, dtype=np.float64)
    out_iters = np.zeros(1, dtype=np.int32)

    t0 = time.time()
    lloyd_c(
        L, None, 400, restarts,
        out_bundles.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_pp_ic.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_shares.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_outside.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_iters.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    elapsed = time.time() - t0

    return out_bundles, out_prices, out_pp_ic[0], elapsed

def evaluate_ot_on_test(test_csv, bundles_indices, prices):
    """Evaluate OT menu on test set"""
    df = pd.read_csv(test_csv)
    wtp_cols = [c for c in df.columns if c.startswith('wtp_')]
    wtps = df[wtp_cols].values
    M, d = wtps.shape

    # Reconstruct bundles from indices
    K = 2 ** d - 1
    bundles_list = []
    for size in range(1, d + 1):
        for combo in combinations(range(d), size):
            bundle = np.zeros(d, dtype=int)
            bundle[list(combo)] = 1
            bundles_list.append(bundle)
    bundles_matrix = np.array(bundles_list)

    # Get actual bundles
    L = len(bundles_indices)
    bundles = np.array([bundles_matrix[int(idx)] for idx in bundles_indices])

    # Compute costs
    c1, c2, b, F = 5.0, 1.0, 2.0, 0.0
    bundle_sizes = bundles.sum(axis=1)
    costs = F + c1 * bundle_sizes + c2 * (bundle_sizes ** b)
    margins = prices - costs

    # Compute utilities for each consumer
    utilities = wtps @ bundles.T - prices.reshape(1, -1)  # (M, L)

    # Add outside option (utility = 0)
    utilities_with_outside = np.concatenate([utilities, np.zeros((M, 1))], axis=1)

    # Each consumer picks best option
    best_choices = np.argmax(utilities_with_outside, axis=1)

    # Compute profit
    chooses_inside = best_choices < L
    chosen_bundles = best_choices[chooses_inside]
    positive_utility = utilities[chooses_inside, chosen_bundles] > 0

    total_profit = np.sum(margins[chosen_bundles][positive_utility])
    pp_ic = total_profit / M

    outside_share = 1.0 - np.sum(utilities.max(axis=1) > 0) / M

    return pp_ic, outside_share

print("="*80)
print("PROPER TRAIN/TEST EVALUATION: Homogeneous Baseline")
print("="*80)
print()

# Create train/test split
csv_path = "scenario_homogeneous_baseline.csv"
print("Creating 80/20 train/test split...")
train_path, test_path, n_train, n_test = create_train_test_split(csv_path, train_ratio=0.8, seed=42)
print(f"  Train: {n_train:,} consumers")
print(f"  Test:  {n_test:,} consumers")
print()

# ============================================================================
# KS EVALUATION
# ============================================================================
print("-"*80)
print("KS (Kohli-Sukumar 1990)")
print("-"*80)

print("Training on train set...")
t0 = time.time()
bundles_ks, prices_ks = ks.train(train_path, L_max=6)
time_train_ks = time.time() - t0
L_ks = len(bundles_ks)

print(f"  Trained L={L_ks} in {time_train_ks:.2f}s")

print("Evaluating on train set...")
pp_train_ks, _ = ks.evaluate(train_path, bundles_ks, prices_ks)
print(f"  Train profit: ${pp_train_ks:.4f}")

print("Evaluating on TEST set...")
pp_test_ks, outside_ks = ks.evaluate(test_path, bundles_ks, prices_ks)
print(f"  TEST profit:  ${pp_test_ks:.4f}")
print(f"  Outside share: {outside_ks:.2%}")
print()

# ============================================================================
# OT EVALUATION
# ============================================================================
print("-"*80)
print("OT (Optimal Transport)")
print("-"*80)

print("Training on train set (trying L=1,2,3,4,5,6)...")
best_L = 1
best_bundles = None
best_prices = None
best_train_profit = 0
best_train_time = 0

for L in range(1, 7):
    bundles, prices, train_profit, elapsed = setup_and_run_ot(train_path, L, restarts=30)
    print(f"  L={L}: Train profit=${train_profit:.4f} in {elapsed:.1f}s")

    if train_profit > best_train_profit:
        best_train_profit = train_profit
        best_L = L
        best_bundles = bundles
        best_prices = prices
        best_train_time = elapsed

print()
print(f"Selected L={best_L} based on train performance")
print(f"  Train profit: ${best_train_profit:.4f}")
print()

print("Evaluating on TEST set...")
pp_test_ot, outside_ot = evaluate_ot_on_test(test_path, best_bundles, best_prices)
print(f"  TEST profit:  ${pp_test_ot:.4f}")
print(f"  Outside share: {outside_ot:.2%}")
print()

# ============================================================================
# BASELINES
# ============================================================================
print("-"*80)
print("BASELINES")
print("-"*80)

# Load test data for baselines
df_test = pd.read_csv(test_path)
wtp_cols = [c for c in df_test.columns if c.startswith('wtp_')]
wtps_test = df_test[wtp_cols].values

print("Uniform Pricing (best single product)...")
pp_uniform, _, _ = uniform_pricing(wtps_test)
print(f"  TEST profit: ${pp_uniform:.4f}")

print("Good/Better/Best (3-tier)...")
pp_gbb, _, _ = good_better_best(wtps_test)
print(f"  TEST profit: ${pp_gbb:.4f}")

print("Personalization Oracle (upper bound)...")
pp_oracle = personalization_oracle(wtps_test)
print(f"  TEST profit: ${pp_oracle:.4f}")
print()

# ============================================================================
# RESULTS
# ============================================================================
print("="*80)
print("FINAL RESULTS (OUT-OF-SAMPLE)")
print("="*80)
print()
print(f"{'Method':<25} {'TEST $':<12}")
print("-"*80)
print(f"{'Uniform Pricing':<25} ${pp_uniform:<11.4f}")
print(f"{'Good/Better/Best':<25} ${pp_gbb:<11.4f}")
print(f"{'KS':<25} ${pp_test_ks:<11.4f}")
print(f"{'OT':<25} ${pp_test_ot:<11.4f}")
print(f"{'Oracle (upper bound)':<25} ${pp_oracle:<11.4f}")
print()
print(f"OT vs KS improvement: {100*(pp_test_ot-pp_test_ks)/pp_test_ks:+.2f}%")
print(f"OT % of oracle:        {100*pp_test_ot/pp_oracle:.1f}%")
print("="*80)

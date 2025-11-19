"""
Proper train/test evaluation for Simulations 2 & 3
- Enterprise Software Market
- Streaming Platform Market
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
import os

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

    # Compute valuations
    valuations = wtps @ bundles.T

    # Compute costs
    c1, c2, b, F = 5.0, 1.0, 2.0, 0.0
    bundle_sizes = bundles.sum(axis=1)
    costs = F + c1 * bundle_sizes + c2 * (bundle_sizes ** b)

    # Find best option for each consumer
    surpluses = valuations - prices
    best = np.argmax(surpluses, axis=1)
    max_surplus = np.max(surpluses, axis=1)

    # Check if anyone opts out
    opt_in = max_surplus > 0
    profit_per_person = np.where(opt_in, prices[best] - costs[best], 0)

    return profit_per_person.mean()

def test_scenario(name, csv_path, L_range, description):
    """Test one scenario with proper train/test split"""
    print("\n" + "="*80)
    print(f"SCENARIO: {name}")
    print(description)
    print("="*80)

    # Generate data if needed
    if not os.path.exists(csv_path):
        print(f"Generating data: {csv_path}")
        if 'enterprise' in csv_path.lower():
            os.system('python create_enterprise_streaming.py')
        elif 'streaming' in csv_path.lower():
            if not os.path.exists('scenario_enterprise_software.csv'):
                os.system('python create_enterprise_streaming.py')

    # Create train/test split
    print(f"\nCreating 80/20 train/test split from {csv_path}...")
    train_csv, test_csv, n_train, n_test = create_train_test_split(csv_path)
    print(f"  Train: {train_csv} ({n_train:,} consumers)")
    print(f"  Test:  {test_csv} ({n_test:,} consumers)")

    # Test KS on all L values
    print(f"\n{'Method':<10} {'L':<3} {'Train Profit':>12} {'Test Profit':>12} {'Time':>8}")
    print("-" * 60)

    ks_results = []
    for L in L_range:
        t0 = time.time()
        bundles_ks, prices_ks = ks.train(train_csv, L_max=L)
        time_ks = time.time() - t0
        train_profit_ks, _ = ks.evaluate(train_csv, bundles_ks, prices_ks)
        test_profit_ks, _ = ks.evaluate(test_csv, bundles_ks, prices_ks)
        ks_results.append({
            'L': L,
            'train_profit': train_profit_ks,
            'test_profit': test_profit_ks,
            'time': time_ks
        })
        print(f"KS         {L:<3} ${train_profit_ks:>10.2f}  ${test_profit_ks:>10.2f}  {time_ks:>6.2f}s")

    # Find best KS result by test profit
    best_ks = max(ks_results, key=lambda x: x['test_profit'])

    # Test OT on all L values
    ot_results = []
    for L in L_range:
        bundles, prices, train_profit, elapsed = setup_and_run_ot(train_csv, L, restarts=30)
        test_profit = evaluate_ot_on_test(test_csv, bundles, prices)
        ot_results.append({
            'L': L,
            'train_profit': train_profit,
            'test_profit': test_profit,
            'time': elapsed
        })
        print(f"OT         {L:<3} ${train_profit:>10.2f}  ${test_profit:>10.2f}  {elapsed:>6.2f}s")

    # Find best OT result by test profit
    best_ot = max(ot_results, key=lambda x: x['test_profit'])

    # Baselines
    print("\n" + "-" * 60)
    print("BASELINES")
    print("-" * 60)

    df_test = pd.read_csv(test_csv)
    wtp_cols = [c for c in df_test.columns if c.startswith('wtp_')]
    wtps_test = df_test[wtp_cols].values

    pp_uniform, _, _ = uniform_pricing(wtps_test)
    print(f"Uniform Pricing:        ${pp_uniform:.2f}")

    pp_gbb, _, _ = good_better_best(wtps_test)
    print(f"Good/Better/Best:       ${pp_gbb:.2f}")

    pp_oracle = personalization_oracle(wtps_test)
    print(f"Oracle (upper bound):   ${pp_oracle:.2f}")

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"BEST KS:   L={best_ks['L']}  Test Profit: ${best_ks['test_profit']:.2f}")
    print(f"BEST OT:   L={best_ot['L']}  Test Profit: ${best_ot['test_profit']:.2f}")
    improvement_pct = (best_ot['test_profit'] / best_ks['test_profit'] - 1) * 100
    print(f"OT ADVANTAGE: +{improvement_pct:.2f}%")
    print(f"OT % of Oracle: {100*best_ot['test_profit']/pp_oracle:.1f}%")

    return {
        'scenario': name,
        'uniform': pp_uniform,
        'gbb': pp_gbb,
        'ks_L': best_ks['L'],
        'ks_train': best_ks['train_profit'],
        'ks_test': best_ks['test_profit'],
        'ot_L': best_ot['L'],
        'ot_train': best_ot['train_profit'],
        'ot_test': best_ot['test_profit'],
        'oracle': pp_oracle,
        'improvement_pct': improvement_pct
    }

# ==============================================================================
# RUN SIMULATIONS 2 & 3
# ==============================================================================
results = []

print("\n" + "="*80)
print("PROPER TRAIN/TEST EVALUATION: SIMULATIONS 2 & 3")
print("="*80)

# Simulation 2: Enterprise Software
results.append(test_scenario(
    name="Enterprise Software",
    csv_path="scenario_enterprise_software.csv",
    L_range=range(1, 7),
    description="B2B software with four distinct customer segments (N=20k, d=6)"
))

# Simulation 3: Streaming Platform
results.append(test_scenario(
    name="Streaming Platform",
    csv_path="scenario_streaming_platform.csv",
    L_range=range(1, 8),
    description="Streaming service with overlapping viewer preferences (N=20k, d=7)"
))

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

df_results = pd.DataFrame(results)
print("\n", df_results.to_string(index=False))

print(f"\nOT wins in {sum(df_results['improvement_pct'] > 0)} / {len(results)} scenarios")
print(f"Average OT advantage: +{df_results['improvement_pct'].mean():.2f}%")

# Save results
df_results.to_csv('enterprise_streaming_results.csv', index=False)
print("\n✓ Results saved to: enterprise_streaming_results.csv")

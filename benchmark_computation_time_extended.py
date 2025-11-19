"""
Extended computational time benchmark: KS vs OT C++
Tests on 5k, 10k, 20k, 50k, 100k, 500k, and 1M consumers
"""
import ctypes
import numpy as np
import pandas as pd
import time
from itertools import combinations
import platform
import ks_from_sim as ks
import matplotlib.pyplot as plt

# Load DLL (cross-platform)
if platform.system() == 'Windows':
    dll = ctypes.CDLL("./ot_core.dll")
elif platform.system() == 'Darwin':  # macOS
    dll = ctypes.CDLL("./ot_core.dylib")
else:  # Linux
    dll = ctypes.CDLL("./ot_core.so")

def create_test_dataset(N, d=6, seed=42):
    """Create a test dataset with N consumers and d features"""
    np.random.seed(seed)

    # Use homogeneous MVN for consistent benchmarking
    mu = np.array([20.0] * d)
    variance = 50.0
    correlation = 0.2
    Sigma = np.full((d, d), variance * correlation)
    np.fill_diagonal(Sigma, variance)

    wtp = np.random.multivariate_normal(mu, Sigma, N)
    wtp = np.maximum(wtp, 5)

    df = pd.DataFrame(wtp, columns=[f'wtp_attr{i}' for i in range(d)])
    path = f'benchmark_N{N}.csv'
    df.to_csv(path, index=False)

    return path

def benchmark_ks(csv_path, L_max=6):
    """Benchmark KS algorithm"""
    t0 = time.time()
    bundles, prices = ks.train(csv_path, L_max=L_max)
    elapsed = time.time() - t0

    pp_train, _ = ks.evaluate(csv_path, bundles, prices)

    return elapsed, len(bundles), pp_train

def benchmark_ot(csv_path, L=6, restarts=30):
    """Benchmark OT C++ algorithm"""
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

    return elapsed, L, out_pp_ic[0]

print("="*80)
print("EXTENDED COMPUTATIONAL TIME BENCHMARK: KS vs OT C++")
print("="*80)
print()
print("Testing dataset sizes: 5k, 10k, 20k, 50k, 100k, 500k, 1M")
print("DGP: Homogeneous multivariate normal (d=6, rho=0.2)")
print("Both methods use L_max=6")
print("OT uses 30 random restarts with OpenMP parallelization (16 threads)")
print()

dataset_sizes = [5000, 10000, 20000, 50000, 100000, 500000, 1000000]
results = []

for N in dataset_sizes:
    print("="*80)
    print(f"N = {N:,} consumers")
    print("="*80)

    # Create dataset
    print(f"Creating dataset...")
    csv_path = create_test_dataset(N, d=6)
    print(f"  Saved to: {csv_path}")
    print()

    # Benchmark KS
    print("Running KS (Kohli-Sukumar 1990)...")
    time_ks, L_ks, profit_ks = benchmark_ks(csv_path, L_max=6)
    print(f"  Time: {time_ks:.2f}s")
    print(f"  L: {L_ks}")
    print(f"  Profit: ${profit_ks:.2f}")
    print()

    # Benchmark OT
    print("Running OT C++ (30 restarts, OpenMP parallel)...")
    time_ot, L_ot, profit_ot = benchmark_ot(csv_path, L=6, restarts=30)
    print(f"  Time: {time_ot:.2f}s")
    print(f"  L: {L_ot}")
    print(f"  Profit: ${profit_ot:.2f}")
    print()

    # Compute speedup
    speedup = time_ot / time_ks
    print(f"Time ratio (OT/KS): {speedup:.2f}x")
    print()

    results.append({
        'N': N,
        'KS_time_s': time_ks,
        'OT_time_s': time_ot,
        'KS_L': L_ks,
        'OT_L': L_ot,
        'KS_profit': profit_ks,
        'OT_profit': profit_ot,
        'time_ratio': speedup
    })

# Summary
print("="*80)
print("SUMMARY: COMPUTATIONAL TIME COMPARISON")
print("="*80)
print()

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print()

# Save results
df_results.to_csv('benchmark_computation_time_extended_results.csv', index=False)
print("Results saved to: benchmark_computation_time_extended_results.csv")
print()

# Analysis
print("="*80)
print("KEY FINDINGS")
print("="*80)
print()

print("1. ABSOLUTE TIME:")
print(f"   KS:  {df_results['KS_time_s'].min():.2f}s (N=5k) to {df_results['KS_time_s'].max():.2f}s (N=1M)")
print(f"   OT:  {df_results['OT_time_s'].min():.2f}s (N=5k) to {df_results['OT_time_s'].max():.2f}s (N=1M)")
print()

print("2. SCALABILITY:")
avg_ks_per_1k = df_results['KS_time_s'].iloc[-1] / (df_results['N'].iloc[-1] / 1000)
avg_ot_per_1k = df_results['OT_time_s'].iloc[-1] / (df_results['N'].iloc[-1] / 1000)
print(f"   KS:  ~{avg_ks_per_1k:.4f}s per 1,000 consumers (at N=1M)")
print(f"   OT:  ~{avg_ot_per_1k:.4f}s per 1,000 consumers (at N=1M)")
print()

print("3. TIME RATIO (OT/KS):")
print(f"   Average: {df_results['time_ratio'].mean():.2f}x")
print(f"   Range:   {df_results['time_ratio'].min():.2f}x to {df_results['time_ratio'].max():.2f}x")
print()

# Create visualization
print("="*80)
print("Creating visualization...")
print("="*80)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Absolute time comparison
ax = axes[0, 0]
ax.plot(df_results['N']/1000, df_results['KS_time_s'], 'o-', linewidth=2, markersize=8, label='KS', color='#2E86AB')
ax.plot(df_results['N']/1000, df_results['OT_time_s'], 's-', linewidth=2, markersize=8, label='OT C++', color='#A23B72')
ax.set_xlabel('Number of Consumers (thousands)', fontsize=12)
ax.set_ylabel('Computation Time (seconds)', fontsize=12)
ax.set_title('Absolute Computation Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# Plot 2: Time ratio
ax = axes[0, 1]
ax.plot(df_results['N']/1000, df_results['time_ratio'], 'D-', linewidth=2, markersize=8, color='#F18F01')
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50x reference')
ax.set_xlabel('Number of Consumers (thousands)', fontsize=12)
ax.set_ylabel('Time Ratio (OT/KS)', fontsize=12)
ax.set_title('Computational Overhead of OT vs KS', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Plot 3: Per-consumer time
ax = axes[1, 0]
ks_per_consumer = df_results['KS_time_s'] / df_results['N'] * 1000000  # microseconds
ot_per_consumer = df_results['OT_time_s'] / df_results['N'] * 1000000  # microseconds
ax.plot(df_results['N']/1000, ks_per_consumer, 'o-', linewidth=2, markersize=8, label='KS', color='#2E86AB')
ax.plot(df_results['N']/1000, ot_per_consumer, 's-', linewidth=2, markersize=8, label='OT C++', color='#A23B72')
ax.set_xlabel('Number of Consumers (thousands)', fontsize=12)
ax.set_ylabel('Time per Consumer (microseconds)', fontsize=12)
ax.set_title('Per-Consumer Computation Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# Plot 4: Scalability comparison
ax = axes[1, 1]
N_vals = df_results['N'].values
ks_times = df_results['KS_time_s'].values
ot_times = df_results['OT_time_s'].values

# Fit linear scaling
ks_rate = ks_times[-1] / N_vals[-1]
ot_rate = ot_times[-1] / N_vals[-1]

ax.scatter(df_results['N']/1000, df_results['KS_time_s'], s=100, label='KS (actual)', color='#2E86AB', alpha=0.6)
ax.scatter(df_results['N']/1000, df_results['OT_time_s'], s=100, label='OT C++ (actual)', color='#A23B72', alpha=0.6)
ax.plot(df_results['N']/1000, N_vals * ks_rate, '--', linewidth=2, label='KS (linear fit)', color='#2E86AB', alpha=0.8)
ax.plot(df_results['N']/1000, N_vals * ot_rate, '--', linewidth=2, label='OT C++ (linear fit)', color='#A23B72', alpha=0.8)
ax.set_xlabel('Number of Consumers (thousands)', fontsize=12)
ax.set_ylabel('Computation Time (seconds)', fontsize=12)
ax.set_title('Linear Scalability Verification', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('computation_time_benchmark.png', dpi=300, bbox_inches='tight')
plt.savefig('computation_time_benchmark.pdf', bbox_inches='tight')
print("Saved figures:")
print("  - computation_time_benchmark.png (high-res)")
print("  - computation_time_benchmark.pdf (vector)")
print()

print("="*80)
print("BENCHMARK COMPLETE")
print("="*80)

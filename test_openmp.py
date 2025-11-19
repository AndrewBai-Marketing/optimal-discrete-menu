"""
Quick test to verify OpenMP parallelization is working
"""
import numpy as np
import pandas as pd
import ot_joint_potentials as ot
import time

# Small dataset for quick testing
np.random.seed(42)
n = 1000
d = 4
wtps = np.random.uniform(5, 20, size=(n, d))
df = pd.DataFrame(wtps, columns=[f'wtp_attr{i}' for i in range(d)])
df.to_csv('test_openmp_data.csv', index=False)

ot.init_environment('test_openmp_data.csv')

print("Testing with different numbers of restarts:")
print("If OpenMP is working, time should scale sub-linearly with restarts")
print()

for restarts in [1, 2, 5, 10]:
    t0 = time.time()
    res, _ = ot.lloyd_menu_multistart(4, restarts=restarts, progress=None)
    elapsed = time.time() - t0
    print(f"Restarts={restarts:2d}: {elapsed:6.2f}s (profit=${res.pp_ic:.4f})")

print()
print("Expected behavior:")
print("  - WITHOUT OpenMP: time doubles when restarts double (linear scaling)")
print("  - WITH OpenMP: time increases much less (sublinear, e.g., 2x restarts = 1.3x time)")

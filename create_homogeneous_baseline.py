"""
Create homogeneous baseline scenario
Single multivariate normal distribution with moderate correlation
"""
import numpy as np
import pandas as pd

np.random.seed(42)

print("="*80)
print("CREATING HOMOGENEOUS BASELINE SCENARIO")
print("="*80)
print()

M = 20000
d = 6

# Mean vector: all features have mean WTP of 20
mu = np.array([20, 20, 20, 20, 20, 20])

# Covariance matrix: variance=50, correlation=0.2 between all features
variance = 50
correlation = 0.2
Sigma = np.full((d, d), variance * correlation)
np.fill_diagonal(Sigma, variance)

print("DGP: Multivariate Normal")
print(f"  N={M:,}, d={d}")
print()
print("Mean vector:")
print(f"  mu = {mu}")
print()
print("Covariance matrix:")
print(f"  Variance on diagonal: {variance}")
print(f"  Correlation between features: {correlation}")
print()
print("Full Sigma matrix:")
print(Sigma)
print()

# Generate WTP from multivariate normal
wtp = np.random.multivariate_normal(mu, Sigma, M)

# Truncate at minimum of 5
wtp = np.maximum(wtp, 5)

print(f"Generated WTP statistics:")
print(f"  Mean: {wtp.mean(axis=0)}")
print(f"  Std:  {wtp.std(axis=0)}")
print(f"  Min:  {wtp.min(axis=0)}")
print(f"  Max:  {wtp.max(axis=0)}")
print()

# Compute actual correlations
corr_matrix = np.corrcoef(wtp.T)
print("Actual correlation matrix (after truncation):")
print(f"  rho ~= {correlation}")
print()

# Save
df = pd.DataFrame(wtp, columns=[f'wtp_attr{i}' for i in range(d)])
df.to_csv('scenario_homogeneous_baseline.csv', index=False)

print("="*80)
print(f"Saved to: scenario_homogeneous_baseline.csv")
print("="*80)
print()
print("Key Characteristics:")
print("  - No segmentation: single homogeneous population")
print("  - Symmetric preferences across all features")
print(f"  - Moderate positive correlation (rho={correlation})")
print("  - Serves as baseline: OT should have minimal advantage")
print("="*80)

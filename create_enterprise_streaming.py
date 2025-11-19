"""
Create data for Simulations 2 & 3 from simulation_results.tex:
- Enterprise Software Market (N=20k, d=6)
- Streaming Platform Market (N=20k, d=7)
"""
import numpy as np
import pandas as pd

np.random.seed(42)

print("="*80)
print("CREATING SIMULATIONS 2 & 3 DATASETS")
print("="*80)

# ==============================================================================
# SIMULATION 2: Enterprise Software Market (N=20k, d=6)
# ==============================================================================
print("\nSIMULATION 2: Enterprise Software Market")
print("-" * 80)

N = 20000
d = 6
features = ['Core', 'Analytics', 'Automation', 'Security', 'API', 'Support']

# Segment sizes (must sum to N)
n_small = 12000      # 60%
n_prof = 4000        # 20%
n_dev = 2400         # 12%
n_ent = 1600         # 8%

consumers = []

# Small Business (60%)
for i in range(n_small):
    base = np.random.uniform(8, 15, d)
    base[0] += 10  # Core
    consumers.append(base)

# Professionals (20%)
for i in range(n_prof):
    base = np.random.uniform(10, 18, d)
    base[0] += 12  # Core
    base[1] += 35  # Analytics
    base[2] += 30  # Automation
    base[4] += 25  # API
    consumers.append(base)

# Developers (12%)
for i in range(n_dev):
    base = np.random.uniform(12, 20, d)
    base[0] += 15  # Core
    base[2] += 40  # Automation
    base[4] += 45  # API
    base[3] += 20  # Security
    consumers.append(base)

# Enterprise (8%)
for i in range(n_ent):
    base = np.random.uniform(15, 25, d)
    base[0] += 20  # Core
    base[3] += 60  # Security
    base[5] += 55  # Support
    base[1] += 35  # Analytics
    consumers.append(base)

# Shuffle
wtps_enterprise = np.array(consumers)
np.random.shuffle(wtps_enterprise)

# Save
df_enterprise = pd.DataFrame(wtps_enterprise, columns=[f'wtp_{feat.lower()}' for feat in features])
df_enterprise.to_csv('scenario_enterprise_software.csv', index=False)

print(f"✓ Created: scenario_enterprise_software.csv")
print(f"  - N = {N:,} consumers")
print(f"  - d = {d} features: {features}")
print(f"  - Segments: Small (60%), Professionals (20%), Developers (12%), Enterprise (8%)")
print(f"  - WTP range: ${wtps_enterprise.min():.1f} to ${wtps_enterprise.max():.1f}")

# ==============================================================================
# SIMULATION 3: Streaming Platform Market (N=20k, d=7)
# ==============================================================================
print("\nSIMULATION 3: Streaming Platform Market")
print("-" * 80)

N = 20000
d = 7
features = ['Movies', 'Series', 'Kids', 'Sports', 'News', 'International', 'Documentaries']

# Step 1: All consumers start with base WTP
wtps_streaming = np.random.uniform(5, 12, (N, d))

# Step 2: Overlapping segment assignment (sampling WITH replacement)
# Families (25%)
n_families = int(0.25 * N)
families_idx = np.random.choice(N, n_families, replace=False)
for i in families_idx:
    wtps_streaming[i, 0] += np.random.uniform(25, 35)  # Movies
    wtps_streaming[i, 2] += np.random.uniform(30, 40)  # Kids
    wtps_streaming[i, 1] += np.random.uniform(15, 25)  # Series

# Sports Fans (18%)
n_sports = int(0.18 * N)
sports_idx = np.random.choice(N, n_sports, replace=False)
for i in sports_idx:
    wtps_streaming[i, 3] += np.random.uniform(40, 50)  # Sports
    wtps_streaming[i, 4] += np.random.uniform(20, 30)  # News

# International Viewers (12%)
n_intl = int(0.12 * N)
intl_idx = np.random.choice(N, n_intl, replace=False)
for i in intl_idx:
    wtps_streaming[i, 5] += np.random.uniform(35, 45)  # International
    wtps_streaming[i, 1] += np.random.uniform(20, 30)  # Series

# Cinephiles (15%)
n_cine = int(0.15 * N)
cine_idx = np.random.choice(N, n_cine, replace=False)
for i in cine_idx:
    wtps_streaming[i, 0] += np.random.uniform(30, 40)  # Movies
    wtps_streaming[i, 6] += np.random.uniform(35, 45)  # Documentaries

# Save
df_streaming = pd.DataFrame(wtps_streaming, columns=[f'wtp_{feat.lower()}' for feat in features])
df_streaming.to_csv('scenario_streaming_platform.csv', index=False)

print(f"✓ Created: scenario_streaming_platform.csv")
print(f"  - N = {N:,} consumers")
print(f"  - d = {d} features: {features}")
print(f"  - Overlapping segments: Families (25%), Sports (18%), International (12%), Cinephiles (15%)")
print(f"  - WTP range: ${wtps_streaming.min():.1f} to ${wtps_streaming.max():.1f}")

print("\n" + "="*80)
print("✓ ALL DATASETS CREATED")
print("="*80)
print("\nNext steps:")
print("1. Run: python test_enterprise_streaming_proper.py")
print("2. Results should match simulation_results.tex:")
print("   - Enterprise: OT beats KS by +9.43%")
print("   - Streaming: OT beats KS by +9.07%")

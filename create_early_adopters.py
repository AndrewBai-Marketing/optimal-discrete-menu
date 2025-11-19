"""
Create larger niche scenario with N=50,000
"""
import numpy as np
import pandas as pd

np.random.seed(42)

print("=" * 80)
print("Creating Niche Scenario with N=50,000")
print("=" * 80)

M = 50000
d = 5

# 95% regular consumers
n_regular = int(M * 0.95)
wtp_regular = np.random.uniform(10, 20, size=(n_regular, d))

# 5% premium consumers who LOVE bundle [1,2,4] (worth 3x)
n_premium = M - n_regular
wtp_premium = np.random.uniform(10, 20, size=(n_premium, d))
wtp_premium[:, 1] += 80
wtp_premium[:, 2] += 80
wtp_premium[:, 4] += 80

wtps_niche = np.vstack([wtp_regular, wtp_premium])
np.random.shuffle(wtps_niche)

df_niche = pd.DataFrame(wtps_niche, columns=[f'wtp_attr{i}' for i in range(d)])
df_niche.to_csv('scenario_niche_50k.csv', index=False)

print(f"Created scenario_niche_50k.csv: {M} consumers, {d} attributes")
print(f"  Regular (95% = {n_regular:,}): Moderate WTP for all attributes")
print(f"  Premium (5% = {n_premium:,}): Extremely high WTP for bundle [1,2,4]")
print(f"  Expected: OT should discover the premium niche")
print("=" * 80)

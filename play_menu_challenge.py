"""
Interactive Menu Design Challenge

Can you design a better menu than the Optimal Transport algorithm?

Try designing bundles and prices for real consumer data, then see how
your profit compares to the OT solution!
"""
import ctypes
import numpy as np
import pandas as pd
import time
from itertools import combinations
import os
import platform

# Load DLL (cross-platform)
try:
    if platform.system() == 'Windows':
        dll = ctypes.CDLL("./ot_core.dll")
    elif platform.system() == 'Darwin':  # macOS
        dll = ctypes.CDLL("./ot_core.dylib")
    else:  # Linux
        dll = ctypes.CDLL("./ot_core.so")
except Exception as e:
    print(f"Error loading compiled library: {e}")
    print(f"Platform: {platform.system()}")
    print("Please ensure the compiled binary exists for your platform.")
    exit(1)

SCENARIOS = {
    1: {
        'name': 'Premium Niche',
        'csv': 'scenario_niche_50k.csv',
        'create_script': 'create_early_adopters.py',
        'd': 5,
        'L': 2,
        'features': ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
        'description': '95% regular consumers, 5% premium segment with high WTP for specific features'
    },
    2: {
        'name': 'Enterprise Software',
        'csv': 'scenario_enterprise_software.csv',
        'create_script': 'create_enterprise_streaming.py',
        'd': 6,
        'L': 3,
        'features': ['Core', 'Analytics', 'Automation', 'Security', 'API', 'Support'],
        'description': '4 business segments: Small (60%), Professionals (20%), Developers (12%), Enterprise (8%)'
    },
    3: {
        'name': 'Streaming Platform',
        'csv': 'scenario_streaming_platform.csv',
        'create_script': 'create_enterprise_streaming.py',
        'd': 7,
        'L': 6,
        'features': ['Movies', 'Series', 'Kids', 'Sports', 'News', 'International', 'Docs'],
        'description': 'Overlapping viewer segments: Families, Sports Fans, International, Cinephiles'
    },
    4: {
        'name': 'Homogeneous Baseline',
        'csv': 'scenario_homogeneous_baseline.csv',
        'create_script': 'create_homogeneous_baseline.py',
        'd': 6,
        'L': 6,
        'features': ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6'],
        'description': 'No clear segments - single multivariate normal distribution'
    }
}

def ensure_data_exists(scenario_info):
    """Generate data if needed"""
    if not os.path.exists(scenario_info['csv']):
        print(f"\nGenerating {scenario_info['name']} data...")
        os.system(f'python {scenario_info["create_script"]}')
        if not os.path.exists(scenario_info['csv']):
            print(f"Error: Could not create {scenario_info['csv']}")
            return False
    return True

def load_scenario(scenario_num, sample_size=1000):
    """Load scenario data with train/test split"""
    info = SCENARIOS[scenario_num]

    if not ensure_data_exists(info):
        return None, None, None

    df = pd.read_csv(info['csv'])

    # Sample for interactive speed
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    # 80/20 train/test split
    train_size = int(0.8 * len(df_sample))
    df_train = df_sample.iloc[:train_size]
    df_test = df_sample.iloc[train_size:]

    return df_train, df_test, info

def print_scenario_info(info):
    """Display scenario details"""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {info['name']}")
    print(f"{'='*70}")
    print(f"{info['description']}")
    print(f"\nFeatures: {', '.join(info['features'])}")
    print(f"Your task: Design a menu with {info['L']} bundles")
    print(f"{'='*70}")

def show_training_data_summary(df_train, features, scenario_name):
    """Show summary statistics from training data and save to CSV"""
    print(f"\n{'='*70}")
    print("TRAINING DATA SUMMARY")
    print(f"{'='*70}")

    wtp_cols = [c for c in df_train.columns if c.startswith('wtp_')]
    wtps = df_train[wtp_cols].values

    print(f"\nFeature WTP Statistics (n={len(df_train)} consumers):")
    print(f"{'Feature':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)

    for i, feat in enumerate(features):
        mean_wtp = wtps[:, i].mean()
        std_wtp = wtps[:, i].std()
        min_wtp = wtps[:, i].min()
        max_wtp = wtps[:, i].max()
        print(f"{feat:<20} ${mean_wtp:>7.2f} ${std_wtp:>7.2f} ${min_wtp:>7.2f} ${max_wtp:>7.2f}")

    # Total WTP statistics
    total_wtp = wtps.sum(axis=1)
    print(f"\n{'Total WTP per consumer':<20} ${total_wtp.mean():>7.2f} ${total_wtp.std():>7.2f} ${total_wtp.min():>7.2f} ${total_wtp.max():>7.2f}")

    # Correlation insights
    corr_matrix = np.corrcoef(wtps.T)
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr_matrix[i,j]) > 0.6:
                high_corr_pairs.append((features[i], features[j], corr_matrix[i,j]))

    if high_corr_pairs:
        print(f"\nHigh Correlations (|r| > 0.6):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  {feat1} <-> {feat2}: r = {corr:.2f}")

    # Save training data to CSV
    csv_filename = f"challenge_train_data_{scenario_name.lower().replace(' ', '_')}.csv"
    df_train.to_csv(csv_filename, index=False)
    print(f"\nTraining data saved to: {csv_filename}")
    print(f"  (You can analyze this in Excel, Python, R, etc.)")

    print(f"{'='*70}")

def get_user_menu(info):
    """Interactive menu design"""
    d = info['d']
    L = info['L']
    features = info['features']

    print(f"\nDESIGN YOUR MENU")
    print(f"You'll create {L} bundles. For each bundle:")
    print(f"  1. Choose which features to include")
    print(f"  2. Set a price")

    bundles = []
    prices = []

    for i in range(L):
        print(f"\n--- Bundle {i+1} of {L} ---")

        # Show features with numbers
        for j, feat in enumerate(features):
            print(f"  {j+1}. {feat}")

        # Get bundle composition
        while True:
            response = input(f"\nEnter feature numbers to include (e.g., '1,2,4' or '1 2 4'): ").strip()

            try:
                # Parse input
                if ',' in response:
                    selected = [int(x.strip()) - 1 for x in response.split(',')]
                else:
                    selected = [int(x.strip()) - 1 for x in response.split()]

                # Validate
                if all(0 <= s < d for s in selected):
                    bundle = np.zeros(d, dtype=int)
                    bundle[selected] = 1
                    bundles.append(bundle)

                    # Show what they selected
                    selected_features = [features[s] for s in selected]
                    print(f"  > Bundle includes: {', '.join(selected_features)}")
                    break
                else:
                    print(f"  X Invalid feature numbers. Use 1-{d}")
            except:
                print(f"  X Invalid input. Try again (e.g., '1,2,4')")

        # Get price
        while True:
            try:
                price = float(input(f"Set price for this bundle: $"))
                if price >= 0:
                    prices.append(price)
                    print(f"  > Price set to ${price:.2f}")
                    break
                else:
                    print("  X Price must be non-negative")
            except:
                print("  X Invalid price. Enter a number (e.g., 50)")

    return np.array(bundles), np.array(prices)

def evaluate_menu(wtps, bundles, prices):
    """Calculate profit for a menu"""
    # Calculate valuations
    valuations = wtps @ bundles.T

    # Calculate costs: C(k) = 5k + k^2
    bundle_sizes = bundles.sum(axis=1)
    costs = 5.0 * bundle_sizes + bundle_sizes ** 2

    # Find best option per consumer
    surpluses = valuations - prices
    best = np.argmax(surpluses, axis=1)
    max_surplus = np.max(surpluses, axis=1)

    # Only count if surplus > 0 (consumer opts in)
    opt_in = max_surplus > 0
    profit_per_person = np.where(opt_in, prices[best] - costs[best], 0)

    participation_rate = opt_in.sum() / len(opt_in) * 100

    return profit_per_person.mean(), participation_rate

def run_ot(wtps, L):
    """Run OT algorithm"""
    M, d = wtps.shape

    # Generate all possible bundles
    K = 2 ** d - 1
    bundles_list = []
    for size in range(1, d + 1):
        for combo in combinations(range(d), size):
            bundle = np.zeros(d, dtype=int)
            bundle[list(combo)] = 1
            bundles_list.append(bundle)

    bundles_matrix = np.array(bundles_list)
    S = wtps @ bundles_matrix.T

    # Costs
    c1, c2, b, F = 5.0, 1.0, 2.0, 0.0
    bundle_sizes = bundles_matrix.sum(axis=1)
    Cx = F + c1 * bundle_sizes + c2 * (bundle_sizes ** b)

    # Initialize C++ environment
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

    # Run OT
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

    lloyd_c(
        L, None, 400, 20,  # 20 restarts for speed
        out_bundles.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_pp_ic.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_shares.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_outside.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_iters.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )

    # Get OT bundles
    ot_bundles = np.array([bundles_matrix[int(idx)] for idx in out_bundles])

    return out_pp_ic[0], ot_bundles, out_prices

def show_results(user_profit, user_participation, ot_profit, ot_bundles, ot_prices, info):
    """Display comparison results"""
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\nYour Menu:")
    print(f"  Profit:        ${user_profit:>8.2f} per person")
    print(f"  Participation: {user_participation:>7.1f}% opted in")

    print(f"\nOT Algorithm:")
    print(f"  Profit:        ${ot_profit:>8.2f} per person")

    print(f"\nOT's Optimal Menu:")
    features = info['features']
    for i, (bundle, price) in enumerate(zip(ot_bundles, ot_prices)):
        selected_features = [features[j] for j in range(len(bundle)) if bundle[j] == 1]
        print(f"  Bundle {i+1}: {', '.join(selected_features)} @ ${price:.2f}")

    gap = (ot_profit - user_profit) / ot_profit * 100

    print(f"\n{'-'*70}")

    if gap < 0:
        print(f"*** INCREDIBLE! You beat OT by {abs(gap):.1f}%!")
        print("    (This shouldn't happen - you found a bug or got very lucky!)")
    elif gap < 2:
        print(f"*** OUTSTANDING! Within {gap:.1f}% of optimal!")
        print("    You have excellent intuition for menu design!")
    elif gap < 10:
        print(f"VERY GOOD! OT profit is {gap:.1f}% higher")
    elif gap < 25:
        print(f"GOOD EFFORT! OT profit is {gap:.1f}% higher")
    else:
        print(f"OT profit is {gap:.1f}% higher")

    print(f"{'-'*70}")

def main():
    """Main interactive loop"""
    print("="*70)
    print("INTERACTIVE MENU DESIGN CHALLENGE")
    print("="*70)
    print("\nCan you design a better menu than the Optimal Transport algorithm?")
    print("Pick bundles and prices for real consumer data, then compare results!")

    while True:
        print("\n" + "="*70)
        print("Choose a scenario:")
        print("="*70)
        for num, info in SCENARIOS.items():
            print(f"{num}. {info['name']:<25} ({info['L']} bundles, {info['d']} features)")
        print("0. Exit")

        try:
            choice = int(input("\nScenario (0-4): "))
            if choice == 0:
                print("\nThanks for playing!")
                break
            if choice not in SCENARIOS:
                print("Invalid choice. Try again.")
                continue
        except:
            print("Invalid input. Try again.")
            continue

        # Load scenario with train/test split
        df_train, df_test, info = load_scenario(choice, sample_size=1000)
        if df_train is None:
            continue

        print_scenario_info(info)

        print(f"\n(Using {len(df_train)} train / {len(df_test)} test consumers)")
        print("You'll design your menu, then both you and OT will be evaluated on held-out test data.")

        # Show training data summary
        show_training_data_summary(df_train, info['features'], info['name'])

        # Get user's menu design
        user_bundles, user_prices = get_user_menu(info)

        # Evaluate user's menu on TEST set
        wtp_cols = [c for c in df_test.columns if c.startswith('wtp_')]
        wtps_test = df_test[wtp_cols].values

        user_profit, user_participation = evaluate_menu(wtps_test, user_bundles, user_prices)

        # Run OT on TRAIN set
        print(f"\nRunning OT algorithm on training data...")
        wtp_cols_train = [c for c in df_train.columns if c.startswith('wtp_')]
        wtps_train = df_train[wtp_cols_train].values

        start = time.time()
        ot_profit_train, ot_bundles, ot_prices = run_ot(wtps_train, info['L'])
        elapsed = time.time() - start
        print(f"> OT training completed in {elapsed:.1f}s")

        # Evaluate OT on TEST set
        ot_profit, _ = evaluate_menu(wtps_test, ot_bundles, ot_prices)
        print(f"> Evaluating on test set...")

        # Show results
        show_results(user_profit, user_participation, ot_profit, ot_bundles, ot_prices, info)

        # Ask to continue
        again = input("\nTry another scenario? (y/n): ")
        if again.lower() != 'y':
            print("\nThanks for playing!")
            break

if __name__ == "__main__":
    main()

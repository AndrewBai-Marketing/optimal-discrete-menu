"""
Simple baseline pricing strategies for comparison
"""
import numpy as np
from itertools import combinations

def uniform_pricing(wtps):
    """
    Best uniform product/price combo: single bundle offered to everyone
    Find the bundle and price that maximizes profit

    Args:
        wtps: N x d matrix of willingness to pay

    Returns:
        profit, bundle, price
    """
    N, d = wtps.shape

    # Try all possible bundles
    best_profit = 0
    best_bundle = None
    best_price = 0

    bundles_list = []
    for size in range(1, d + 1):
        for combo in combinations(range(d), size):
            bundle = np.zeros(d, dtype=int)
            bundle[list(combo)] = 1
            bundles_list.append(bundle)

    for bundle in bundles_list:
        # Calculate each consumer's WTP for this bundle
        bundle_wtp = wtps[:, bundle == 1].sum(axis=1)

        # Cost of this bundle: C(k) = 5k + k^2
        k = bundle.sum()
        cost = 5 * k + k ** 2

        # Try different prices (use quantiles of bundle WTP)
        prices_to_try = np.percentile(bundle_wtp, np.arange(10, 100, 5))

        for price in prices_to_try:
            # Who buys?
            buyers = bundle_wtp >= price
            profit = buyers.sum() * (price - cost) / N

            if profit > best_profit:
                best_profit = profit
                best_bundle = bundle.copy()
                best_price = price

    return best_profit, best_bundle, best_price


def good_better_best(wtps):
    """
    Good/Better/Best tiered pricing: 3 bundles of increasing size
    Simple marketing manager strategy: small/medium/large bundles

    Args:
        wtps: N x d matrix of willingness to pay

    Returns:
        profit, bundles (list of 3), prices (list of 3)
    """
    N, d = wtps.shape

    # Strategy: Good = 2 features, Better = d/2 features, Best = all features
    # Pick features with highest average WTP
    avg_wtp = wtps.mean(axis=0)
    sorted_features = np.argsort(avg_wtp)[::-1]

    # Good: top 2 features
    good_bundle = np.zeros(d, dtype=int)
    good_bundle[sorted_features[:2]] = 1

    # Better: top d//2 features
    better_bundle = np.zeros(d, dtype=int)
    better_bundle[sorted_features[:max(3, d//2)]] = 1

    # Best: all features
    best_bundle = np.ones(d, dtype=int)

    bundles = [good_bundle, better_bundle, best_bundle]

    # Calculate costs
    costs = []
    for bundle in bundles:
        k = bundle.sum()
        costs.append(5 * k + k ** 2)

    # Calculate each consumer's WTP for each bundle
    bundle_wtps = []
    for bundle in bundles:
        bundle_wtp = wtps[:, bundle == 1].sum(axis=1)
        bundle_wtps.append(bundle_wtp)

    # Optimize prices jointly (simple grid search)
    best_profit = 0
    best_prices = [0, 0, 0]

    # Try percentile-based pricing
    for p1_pct in range(20, 80, 10):  # Good price
        for p2_pct in range(30, 90, 10):  # Better price
            for p3_pct in range(40, 95, 10):  # Best price
                prices = [
                    np.percentile(bundle_wtps[0], p1_pct),
                    np.percentile(bundle_wtps[1], p2_pct),
                    np.percentile(bundle_wtps[2], p3_pct)
                ]

                # Each consumer picks best option (including outside option)
                utilities = np.zeros((N, 4))  # 3 bundles + outside option
                for i, (bwtp, price) in enumerate(zip(bundle_wtps, prices)):
                    utilities[:, i] = bwtp - price
                # utilities[:, 3] is already 0 (outside option)

                choices = utilities.argmax(axis=1)

                profit = 0
                for i in range(3):
                    profit += ((choices == i).sum() * (prices[i] - costs[i])) / N

                if profit > best_profit:
                    best_profit = profit
                    best_prices = prices.copy()

    return best_profit, bundles, best_prices


def personalization_oracle(wtps):
    """
    Upper bound: perfect personalization
    For each consumer, offer the bundle and price that maximizes profit

    Args:
        wtps: N x d matrix of willingness to pay

    Returns:
        profit (average per consumer)
    """
    N, d = wtps.shape

    total_profit = 0

    # For each consumer
    for i in range(N):
        consumer_wtp = wtps[i]

        best_profit = 0  # Outside option

        # Try all bundles
        bundles_list = []
        for size in range(1, d + 1):
            for combo in combinations(range(d), size):
                bundle = np.zeros(d, dtype=int)
                bundle[list(combo)] = 1
                bundles_list.append(bundle)

        for bundle in bundles_list:
            # Consumer's WTP for this bundle
            bundle_wtp = consumer_wtp[bundle == 1].sum()

            # Cost
            k = bundle.sum()
            cost = 5 * k + k ** 2

            # Optimal price: charge exactly their WTP
            price = bundle_wtp

            # Profit if consumer buys
            profit = price - cost

            # Only profitable if they would buy and profit > 0
            if profit > best_profit:
                best_profit = profit

        total_profit += best_profit

    return total_profit / N

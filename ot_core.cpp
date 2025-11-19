// ot_core.cpp - C++ implementation of optimal transport Lloyd algorithm
// Optimized core for menu pricing with IC constraints
//
// Build: g++ -O3 -march=native -fPIC -shared -std=c++17 -I/path/to/eigen3 -o ot_core.so ot_core.cpp
// Or with pybind11: See build instructions below

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <random>
#include <iostream>
#include <map>
#include <set>

using namespace Eigen;
using namespace std;

// Global state (matches Python implementation)
static MatrixXd S;           // (M, K) - valuations
static VectorXd Cx;          // (K,) - costs
static int M = 0;            // num customers
static int K = 0;            // num bundles in library
static double cF = 0.0;      // fixed cost
static double c1 = 5.0;      // linear cost
static double c2 = 1.5;      // convex cost
static double b = 2.0;       // convex exponent

struct MenuResult {
    VectorXi bundles;        // (L,) bundle indices
    VectorXd prices;         // (L,) prices
    double pp_ic;            // per-person profit
    VectorXd shares;         // (L,) market shares
    double outside;          // outside option share
    int iters;               // iterations used
};

// Cost function matching Python: c(bundle) = c1 * sum(bundle)^b + c2 * num_attrs
inline double bundle_cost(int bundle_idx) {
    return Cx(bundle_idx);
}

// Evaluate IC profit for a menu
MenuResult eval_ic(const VectorXi& bundles, const VectorXd& prices) {
    int L = bundles.size();
    MenuResult result;
    result.bundles = bundles;
    result.prices = prices;
    result.shares = VectorXd::Zero(L);
    result.iters = 0;

    // Compute surplus for each customer-bundle pair
    MatrixXd surplus(M, L);
    for (int l = 0; l < L; ++l) {
        surplus.col(l) = S.col(bundles(l)) - VectorXd::Constant(M, prices(l));
    }

    // Find best option for each customer
    double profit_sum = 0.0;
    int outside_count = 0;

    for (int i = 0; i < M; ++i) {
        int best_j = -1;
        double best_s = 0.0;  // Outside option gives 0 surplus

        for (int l = 0; l < L; ++l) {
            if (surplus(i, l) > best_s) {
                best_s = surplus(i, l);
                best_j = l;
            }
        }

        if (best_j >= 0) {
            result.shares(best_j) += 1.0;
            profit_sum += (prices(best_j) - Cx(bundles(best_j)));
        } else {
            outside_count++;
        }
    }

    result.shares /= M;
    result.pp_ic = profit_sum / M;
    result.outside = static_cast<double>(outside_count) / M;

    return result;
}

// Best response pricing for one slot, accounting for cannibalization
// This matches Python's update_price_for_segment_ic() logic
pair<double, double> best_price_given_alt(const VectorXd& Vj, double C, const VectorXd& alt,
                                          const VectorXi& bundles, const VectorXd& prices, int slot_l) {
    // VERSION MARKER - Always write this to prove DLL is loaded
    FILE* version_marker = fopen("dll_version.txt", "a");
    if (version_marker) {
        fprintf(version_marker, "DLL_VERSION: 2024-11-15-FIXED-V2 slot=%d\n", slot_l);
        fclose(version_marker);
    }

    // DEBUG: Trace EVERY call during iteration 0
    static int global_call_count = 0;
    global_call_count++;
    bool do_trace = (slot_l == 0 && prices.sum() < 1e-9);  // initialization
    bool do_iter_trace = (global_call_count >= 3 && global_call_count <= 8);  // calls 3-8 are iteration 0
    FILE* debug_file = nullptr;

    if (do_trace) {
        debug_file = fopen("cpp_pricing_trace.txt", "w");
        if (debug_file) {
            fprintf(debug_file, "\n=== C++ PRICING TRACE - Slot %d, Bundle %d ===\n", slot_l, bundles(slot_l));
            fprintf(debug_file, "Cost: %.6f\n", C);
        }
    } else if (do_iter_trace) {
        debug_file = fopen("cpp_pricing_trace.txt", "a");
        if (debug_file) {
            fprintf(debug_file, "\n=== ITER0 CALL %d - Slot %d, Bundle %d ===\n", global_call_count, slot_l, bundles(slot_l));
            fprintf(debug_file, "Cost: %.6f\n", C);
            fprintf(debug_file, "Current prices: [%.6f, %.6f]\n", prices(0), prices(1));
        }
    }

    // Threshold pricing: customer i buys if p <= Vj(i) - alt(i)
    VectorXd T = Vj - alt;

    // Compute lost margins (cannibalization cost)
    // lost_margin[i] = margin lost if customer i switches FROM their current best alternative TO slot_l
    // Note: alt[i] is already the utility of customer i's best alternative (excluding slot_l)
    VectorXd lost_margin = VectorXd::Zero(M);

    int L = bundles.size();

    // DEBUG: Track alternatives before clipping
    if (do_trace && debug_file) {
        fprintf(debug_file, "\n1. Alternatives (first 10 customers):\n");
        for (int i = 0; i < min(10, (int)M); ++i) {
            fprintf(debug_file, "   Customer %d: alt[%d] = %.6f (before clip)\n", i, i, alt(i));
        }
    }

    // For each customer, find which bundle is their best alternative and compute its margin
    int num_pos_alt = 0;
    for (int i = 0; i < M; ++i) {
        // alt[i] is the utility of best alternative
        // We need to find WHICH bundle it is to compute the margin
        double alt_val = max(alt(i), 0.0);  // Clip to outside option

        if (alt_val > 0.0) {
            num_pos_alt++;
            // Find which bundle gives this utility
            int best_alt_idx = -1;
            for (int ll = 0; ll < L; ++ll) {
                if (ll == slot_l) continue;
                double util = S(i, bundles(ll)) - prices(ll);
                if (abs(util - alt_val) < 1e-9) {  // Match found
                    best_alt_idx = ll;
                    break;
                }
            }

            if (best_alt_idx >= 0) {
                lost_margin(i) = prices(best_alt_idx) - Cx(bundles(best_alt_idx));
            }
        }
    }

    // DEBUG: Lost margins
    if (do_trace && debug_file) {
        fprintf(debug_file, "\n2. Lost margins:\n");
        fprintf(debug_file, "   Number with positive alt: %d\n", num_pos_alt);
        fprintf(debug_file, "   First 10 lost_margin values:\n");
        for (int i = 0; i < min(10, (int)M); ++i) {
            fprintf(debug_file, "      lost_margin[%d] = %.6f\n", i, lost_margin(i));
        }
    }

    // Filter to profitable customers (T > C)
    vector<int> valid_indices;
    vector<double> T_valid, LM_valid;
    for (int i = 0; i < M; ++i) {
        if (T(i) > C + 1e-9) {
            valid_indices.push_back(i);
            T_valid.push_back(T(i));
            LM_valid.push_back(lost_margin(i));
        }
    }

    // DEBUG: Threshold computation
    if ((do_trace || do_iter_trace) && debug_file) {
        fprintf(debug_file, "\n3. Thresholds (first 10 customers):\n");
        for (int i = 0; i < min(10, (int)M); ++i) {
            fprintf(debug_file, "   Vj[%d] = %.6f, T[%d] = %.6f\n", i, Vj(i), i, T(i));
        }
        fprintf(debug_file, "\n4. Profitable customers: %zu / %d\n", T_valid.size(), (int)M);
    }

    if (T_valid.empty()) {
        return {C + 1e-9, 0.0};
    }

    // Sort by threshold descending
    vector<size_t> order(T_valid.size());
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&T_valid](size_t a, size_t b) {
        return T_valid[a] > T_valid[b];
    });

    // DEBUG: Sorted candidates
    if (do_trace && debug_file) {
        fprintf(debug_file, "\n5. Sorted candidates (first 10):\n");
        for (size_t k = 0; k < min((size_t)10, order.size()); ++k) {
            size_t idx = order[k];
            fprintf(debug_file, "   k=%zu: T=%.6f, LM=%.6f\n", k, T_valid[idx], LM_valid[idx]);
        }
    }

    // Try each threshold, accounting for cannibalization
    double best_p = C + 1e-9;
    double best_profit = 0.0;
    int best_k = -1;

    double cum_lost = 0.0;
    for (size_t k = 0; k < order.size(); ++k) {
        size_t idx = order[k];
        double p = T_valid[idx] * (1.0 - 1e-10);  // Just below threshold
        if (p <= C) continue;

        cum_lost += LM_valid[idx];

        // Total menu profit if we set price to p:
        // gain (k+1) buyers at margin (p-C), but lose cum_lost in cannibalized margins
        double margin = p - C;
        int buyers = static_cast<int>(k + 1);
        double profit = (margin * buyers - cum_lost) / M;

        if (profit > best_profit) {
            best_profit = profit;
            best_p = p;
            best_k = (int)k;
        }
    }

    // DEBUG: Result
    if ((do_trace || do_iter_trace) && debug_file) {
        fprintf(debug_file, "\n6. RESULT:\n");
        fprintf(debug_file, "   Best k: %d\n", best_k);
        fprintf(debug_file, "   Best price: %.10f\n", best_p);
        fprintf(debug_file, "   Best profit: %.10f\n", best_profit);
        if (best_k >= 0) {
            size_t idx = order[best_k];
            double cum_test = 0.0;
            for (int kk = 0; kk <= best_k; ++kk) {
                cum_test += LM_valid[order[kk]];
            }
            fprintf(debug_file, "   Buyers at this price: %d\n", best_k + 1);
            fprintf(debug_file, "   Cumulative lost margin: %.10f\n", cum_test);
        }
        fprintf(debug_file, "\n=== End C++ trace ===\n");
        fclose(debug_file);
    }

    return {best_p, best_profit};
}

// Compute alternatives excluding each bundle
MatrixXd compute_alt_excluding_each(const MatrixXd& surplus) {
    int L = surplus.cols();
    MatrixXd alt_excl(M, L);

    for (int i = 0; i < M; ++i) {
        // Find top 2 for each customer
        double top1 = -numeric_limits<double>::infinity();
        double top2 = -numeric_limits<double>::infinity();
        int arg1 = 0;

        for (int l = 0; l < L; ++l) {
            if (surplus(i, l) > top1) {
                top2 = top1;
                top1 = surplus(i, l);
                arg1 = l;
            } else if (surplus(i, l) > top2) {
                top2 = surplus(i, l);
            }
        }

        // Alternative for each slot
        for (int l = 0; l < L; ++l) {
            alt_excl(i, l) = (l == arg1) ? max(0.0, top2) : max(0.0, top1);
        }
    }

    return alt_excl;
}

// Lloyd's algorithm - one run
MenuResult lloyd_menu(int L, const VectorXi& init_bundles, int max_outer = 400, double tol = 1e-5) {
    // VERSION MARKER - Prove lloyd_menu is being called
    // FILE* version_lloyd = fopen("lloyd_called.txt", "w");
    // if (version_lloyd) {
    //     fprintf(version_lloyd, "lloyd_menu called: L=%d, max_outer=%d\n", L, max_outer);
    //     fclose(version_lloyd);
    // }

    // DEBUG: Track bundle changes - COMMENTED OUT (causes exponential slowdown)
    // FILE* bundle_trace = fopen("bundle_trace.txt", "w");
    // if (bundle_trace) {
    //     fprintf(bundle_trace, "=== BUNDLE TRACE ===\n");
    //     fprintf(bundle_trace, "init_bundles: [");
    //     for (int l = 0; l < L; ++l) {
    //         fprintf(bundle_trace, "%d%s", init_bundles(l), (l < L-1) ? ", " : "");
    //     }
    //     fprintf(bundle_trace, "]\n");
    //     fclose(bundle_trace);
    // }

    VectorXi bundles = init_bundles;

    // DEBUG: Verify copy - COMMENTED OUT (causes exponential slowdown)
    // bundle_trace = fopen("bundle_trace.txt", "a");
    // if (bundle_trace) {
    //     fprintf(bundle_trace, "After copy to bundles: [");
    //     for (int l = 0; l < L; ++l) {
    //         fprintf(bundle_trace, "%d%s", bundles(l), (l < L-1) ? ", " : "");
    //     }
    //     fprintf(bundle_trace, "]\n");
    //     fclose(bundle_trace);
    // }

    VectorXd prices = VectorXd::Zero(L);

    // Initialize prices (with cannibalization)
    for (int l = 0; l < L; ++l) {
        // BUGFIX: Vj should be UTILITIES not raw valuations!
        // Python computes: V = S[:, bundles] - prices, then uses V[:, j] as Vj
        VectorXd Vj = S.col(bundles(l)) - VectorXd::Constant(M, prices(l));
        VectorXd alt = VectorXd::Zero(M);

        if (L > 1) {
            // Compute alternatives from other bundles
            MatrixXd surplus(M, L);
            for (int ll = 0; ll < L; ++ll) {
                surplus.col(ll) = S.col(bundles(ll)) - VectorXd::Constant(M, prices(ll));
            }
            MatrixXd alt_excl = compute_alt_excluding_each(surplus);
            alt = alt_excl.col(l);
        }

        auto [p, _] = best_price_given_alt(Vj, Cx(bundles(l)), alt, bundles, prices, l);
        prices(l) = p;

        // DEBUG: Check bundles after each pricing iteration - COMMENTED OUT
        // bundle_trace = fopen("bundle_trace.txt", "a");
        // if (bundle_trace) {
        //     fprintf(bundle_trace, "After pricing slot %d: bundles = [", l);
        //     for (int ll = 0; ll < L; ++ll) {
        //         fprintf(bundle_trace, "%d%s", bundles(ll), (ll < L-1) ? ", " : "");
        //     }
        //     fprintf(bundle_trace, "], price[%d] = %.6f\n", l, prices(l));
        //     fclose(bundle_trace);
        // }
    }

    // DEBUG: Before eval_ic - COMMENTED OUT
    // bundle_trace = fopen("bundle_trace.txt", "a");
    // if (bundle_trace) {
    //     fprintf(bundle_trace, "Before eval_ic: bundles = [");
    //     for (int l = 0; l < L; ++l) {
    //         fprintf(bundle_trace, "%d%s", bundles(l), (l < L-1) ? ", " : "");
    //     }
    //     fprintf(bundle_trace, "]\n");
    //     fclose(bundle_trace);
    // }

    MenuResult current = eval_ic(bundles, prices);

    // DEBUG: After eval_ic - COMMENTED OUT
    // bundle_trace = fopen("bundle_trace.txt", "a");
    // if (bundle_trace) {
    //     fprintf(bundle_trace, "After eval_ic: current.bundles = [");
    //     for (int l = 0; l < L; ++l) {
    //         fprintf(bundle_trace, "%d%s", current.bundles(l), (l < L-1) ? ", " : "");
    //     }
    //     fprintf(bundle_trace, "], pp_ic = %.10f\n", current.pp_ic);
    //     fclose(bundle_trace);
    // }

    // VERSION MARKER: Confirm new DLL is loading
    fprintf(stderr, "=== OT_CORE DLL VERSION: WITH_BUNDLE_COORDINATE_ASCENT ===\n");

    // Coordinate ascent
    for (int iter = 0; iter < max_outer; ++iter) {
        double pp_prev = current.pp_ic;

        // DEBUG: Log iteration start - COMMENTED OUT
        // bundle_trace = fopen("bundle_trace.txt", "a");
        // if (bundle_trace) {
        //     fprintf(bundle_trace, "\n--- Iteration %d ---\n", iter);
        //     fprintf(bundle_trace, "pp_prev = %.10f\n", pp_prev);
        //     fprintf(bundle_trace, "prices_before_sweeps = [%.6f, %.6f]\n", prices(0), prices(1));
        //     fclose(bundle_trace);
        // }

        // Price updates (exact best response with cannibalization)
        // IMPORTANT: Update all prices simultaneously, not sequentially!
        // Python does: new_prices[l] = update(..., prices, l) for all l, then prices = new_prices
        for (int sweep = 0; sweep < 3; ++sweep) {
            VectorXd new_prices(L);
            for (int l = 0; l < L; ++l) {
                MatrixXd surplus(M, L);
                for (int ll = 0; ll < L; ++ll) {
                    surplus.col(ll) = S.col(bundles(ll)) - VectorXd::Constant(M, prices(ll));
                }

                MatrixXd alt_excl = compute_alt_excluding_each(surplus);
                // BUGFIX: Pass RAW VALUATION (not surplus) to match initialization behavior
                // During init, prices=0 so S-prices=S. During ascent, we need to pass S too.
                auto [p_star, _] = best_price_given_alt(S.col(bundles(l)), Cx(bundles(l)), alt_excl.col(l),
                                                        bundles, prices, l);
                new_prices(l) = p_star;
            }
            // Update all prices at once
            prices = new_prices;
        }

        // DEBUG: Log after sweeps - COMMENTED OUT
        // bundle_trace = fopen("bundle_trace.txt", "a");
        // if (bundle_trace) {
        //     fprintf(bundle_trace, "prices_after_sweeps = [%.6f, %.6f]\n", prices(0), prices(1));
        //     fclose(bundle_trace);
        // }

        // (C) Bundle coordinate ascent - try replacing each bundle
        double pp0 = eval_ic(bundles, prices).pp_ic;

        // bundle_trace = fopen("bundle_trace.txt", "a");
        // if (bundle_trace) {
        //     fprintf(bundle_trace, "BUNDLE OPTIMIZATION: Starting with pp0 = %.10f\n", pp0);
        //     fclose(bundle_trace);
        // }

        for (int l = 0; l < L; ++l) {
            // Screen candidates: rank by solo profit and try top 200
            int best_j = bundles(l);
            double best_pp = pp0;
            double best_p = prices(l);

            // bundle_trace = fopen("bundle_trace.txt", "a");
            // if (bundle_trace) {
            //     fprintf(bundle_trace, "  Slot %d: Current bundle %d, pp=%.10f\n", l, bundles(l), pp0);
            //     fclose(bundle_trace);
            // }

            // Screen candidates: find bundles not in menu
            set<int> in_menu;
            for (int ll = 0; ll < L; ++ll) {
                in_menu.insert(bundles(ll));
            }

            // Rank all candidates by solo profit (value - cost)
            vector<pair<double, int>> ranked_candidates;
            for (int j = 0; j < K; ++j) {
                if (in_menu.count(j) > 0) continue;
                // Simple screening: solo profit = sum(max(0, valuations - cost)) / M
                VectorXd vals = S.col(j);
                double solo_profit = ((vals.array() - Cx(j)).max(0.0)).sum() / M;
                ranked_candidates.push_back({solo_profit, j});
            }

            // Sort by solo profit (descending) and take top 8 (finding quality threshold)
            sort(ranked_candidates.begin(), ranked_candidates.end(),
                 [](const pair<double,int>& a, const pair<double,int>& b) { return a.first > b.first; });
            int n_cand = min(8, (int)ranked_candidates.size());

            int tries = 0;
            int no_improv_count = 0;
            for (int idx = 0; idx < n_cand; ++idx) {
                int j = ranked_candidates[idx].second;

                // Try replacing bundles[l] with j
                VectorXi trial_bundles = bundles;
                trial_bundles(l) = j;

                // Re-price slot l with new bundle
                MatrixXd surplus(M, L);
                for (int ll = 0; ll < L; ++ll) {
                    surplus.col(ll) = S.col(trial_bundles(ll)) - VectorXd::Constant(M, prices(ll));
                }
                MatrixXd alt_excl = compute_alt_excluding_each(surplus);
                auto [p_try, _] = best_price_given_alt(S.col(j), Cx(j), alt_excl.col(l),
                                                       trial_bundles, prices, l);

                VectorXd trial_prices = prices;
                trial_prices(l) = p_try;

                double pp_try = eval_ic(trial_bundles, trial_prices).pp_ic;

                // Log first few tries - COMMENTED OUT
                // if (tries < 3) {
                //     bundle_trace = fopen("bundle_trace.txt", "a");
                //     if (bundle_trace) {
                //         fprintf(bundle_trace, "    Try j=%d: pp=%.10f vs best=%.10f, improv=%s\n",
                //                 j, pp_try, best_pp, (pp_try > best_pp + 1e-12) ? "YES" : "NO");
                //         fclose(bundle_trace);
                //     }
                // }
                tries++;

                if (pp_try > best_pp + 1e-12) {
                    best_pp = pp_try;
                    best_j = j;
                    best_p = p_try;
                    no_improv_count = 0;
                } else {
                    no_improv_count++;
                    if (no_improv_count >= 15) break;  // Early stopping like Python
                }
            }

            // bundle_trace = fopen("bundle_trace.txt", "a");
            // if (bundle_trace) {
            //     fprintf(bundle_trace, "    Slot %d: Tried %d candidates, best_j=%d\n", l, tries, best_j);
            //     fclose(bundle_trace);
            // }

            // If found better bundle, update
            if (best_j != bundles(l)) {
                fprintf(stderr, "BUNDLE REPLACEMENT: slot %d: bundle %d -> %d (pp: %.4f -> %.4f)\n",
                        l, bundles(l), best_j, pp0, best_pp);
                bundles(l) = best_j;
                prices(l) = best_p;
                pp0 = best_pp;
            }
        }

        current = eval_ic(bundles, prices);

        // DEBUG: Log after eval_ic - COMMENTED OUT
        // bundle_trace = fopen("bundle_trace.txt", "a");
        // if (bundle_trace) {
        //     fprintf(bundle_trace, "pp_after_eval = %.10f\n", current.pp_ic);
        //     fprintf(bundle_trace, "improvement = %.10f\n", current.pp_ic - pp_prev);
        //     fclose(bundle_trace);
        // }

        // Convergence check
        if (current.pp_ic - pp_prev < tol) {
            current.iters = iter + 1;

            // bundle_trace = fopen("bundle_trace.txt", "a");
            // if (bundle_trace) {
            //     fprintf(bundle_trace, "CONVERGED at iteration %d\n", iter);
            //     fclose(bundle_trace);
            // }
            break;
        }

        current.iters = iter + 1;
    }

    return current;
}

// Seed menu with top-L mode bundles (matches Python's seed_menu_modes)
VectorXi seed_menu_modes(int L) {
    // Find most popular bundle for each consumer
    VectorXi best_bundles(M);
    for (int i = 0; i < M; ++i) {
        int best_j = 0;
        double best_val = S(i, 0) - Cx(0);
        for (int j = 1; j < K; ++j) {
            double val = S(i, j) - Cx(j);
            if (val > best_val) {
                best_val = val;
                best_j = j;
            }
        }
        best_bundles(i) = best_j;
    }

    // Count occurrences of each bundle
    map<int, int> counts;
    for (int i = 0; i < M; ++i) {
        counts[best_bundles(i)]++;
    }

    // Sort by count (descending) and take top L
    vector<pair<int, int>> count_vec(counts.begin(), counts.end());
    sort(count_vec.begin(), count_vec.end(),
         [](const pair<int, int>& a, const pair<int, int>& b) {
             return a.second > b.second;  // Sort by count descending
         });

    VectorXi modes(L);
    for (int l = 0; l < min(L, (int)count_vec.size()); ++l) {
        modes(l) = count_vec[l].first;
    }

    // If fewer unique modes than L, fill with random bundles
    if ((int)count_vec.size() < L) {
        mt19937 rng_fill(123);
        vector<int> all_indices(K);
        iota(all_indices.begin(), all_indices.end(), 0);
        shuffle(all_indices.begin(), all_indices.end(), rng_fill);

        int fill_idx = 0;
        for (int l = count_vec.size(); l < L; ++l) {
            // Find a bundle not already in modes
            while (fill_idx < K) {
                bool already_in = false;
                for (int m = 0; m < l; ++m) {
                    if (modes(m) == all_indices[fill_idx]) {
                        already_in = true;
                        break;
                    }
                }
                if (!already_in) {
                    modes(l) = all_indices[fill_idx];
                    fill_idx++;
                    break;
                }
                fill_idx++;
            }
        }
    }

    return modes;
}

// Multi-start wrapper with mode-based initialization (matches Python)
// Parallelized with OpenMP for faster execution
MenuResult lloyd_menu_multistart(int L, int restarts = 3, int max_outer = 400) {
    // Pre-generate all initializations with proper seeds
    vector<VectorXi> all_inits(restarts);

    for (int r = 0; r < restarts; ++r) {
        mt19937 rng(123 + 97 * L + r);  // Unique seed per restart

        VectorXi init_bundles = seed_menu_modes(L);

        // Jitter: randomly swap about 1/3 of bundles (matches Python)
        if (L > 1) {
            int num_swaps = max(1, L / 3);
            vector<int> swap_positions(L);
            iota(swap_positions.begin(), swap_positions.end(), 0);
            shuffle(swap_positions.begin(), swap_positions.end(), rng);

            uniform_int_distribution<int> bundle_dist(0, K - 1);
            for (int i = 0; i < num_swaps; ++i) {
                init_bundles(swap_positions[i]) = bundle_dist(rng);
            }
        }

        all_inits[r] = init_bundles;
    }

    // Parallel execution of all restarts with OpenMP
    vector<MenuResult> results(restarts);

#ifdef _OPENMP
    fprintf(stderr, "=== MULTISTART VERSION: PARALLEL WITH OPENMP (%d threads) ===\n", omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < restarts; ++r) {
        #pragma omp critical
        fprintf(stderr, "Starting restart %d/%d\n", r+1, restarts);

        results[r] = lloyd_menu(L, all_inits[r], max_outer);

        #pragma omp critical
        fprintf(stderr, "Completed restart %d/%d: profit=%.4f\n", r+1, restarts, results[r].pp_ic);
    }
#else
    fprintf(stderr, "=== MULTISTART VERSION: SEQUENTIAL (OPENMP NOT AVAILABLE) ===\n");

    for (int r = 0; r < restarts; ++r) {
        fprintf(stderr, "Starting restart %d/%d\n", r+1, restarts);
        results[r] = lloyd_menu(L, all_inits[r], max_outer);
        fprintf(stderr, "Completed restart %d/%d: profit=%.4f\n", r+1, restarts, results[r].pp_ic);
    }
#endif

    // Find best result
    MenuResult best;
    best.pp_ic = -numeric_limits<double>::infinity();

    for (int r = 0; r < restarts; ++r) {
        if (results[r].pp_ic > best.pp_ic) {
            best = results[r];
        }
    }

    return best;
}

// ==================== Python Interface (extern "C" for simple ctypes) ====================

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

EXPORT void init_environment(const double* S_data, const double* Cx_data,
                             int M_in, int K_in,
                             double cF_in, double c1_in, double c2_in, double b_in) {
    M = M_in;
    K = K_in;
    cF = cF_in;
    c1 = c1_in;
    c2 = c2_in;
    b = b_in;

    // Copy data
    // IMPORTANT: Python sends S flattened with order='C' (row-major)
    // So we need to use RowMajor layout in Eigen
    S = Map<const Matrix<double, Dynamic, Dynamic, RowMajor>>(S_data, M, K);
    Cx = Map<const VectorXd>(Cx_data, K);
}

EXPORT void lloyd_menu_c(int L, const int* init_bundles_data,
                         int max_outer, int restarts,
                         // Output parameters
                         int* out_bundles, double* out_prices,
                         double* out_pp_ic, double* out_shares, double* out_outside, int* out_iters) {

    // VERSION MARKER - Prove lloyd_menu_c is being called
    FILE* version_c = fopen("lloyd_c_called.txt", "w");
    if (version_c) {
        fprintf(version_c, "lloyd_menu_c called: L=%d, max_outer=%d, restarts=%d, has_init=%d\n",
                L, max_outer, restarts, (init_bundles_data != nullptr));
        fclose(version_c);
    }

    MenuResult result;

    if (init_bundles_data != nullptr && restarts == 1) {
        // Single run with provided initialization
        VectorXi init_bundles = Map<const VectorXi>(init_bundles_data, L);
        result = lloyd_menu(L, init_bundles, max_outer);
    } else {
        // Multi-start
        result = lloyd_menu_multistart(L, restarts, max_outer);
    }

    // Copy results to output arrays
    for (int l = 0; l < L; ++l) {
        out_bundles[l] = result.bundles(l);
        out_prices[l] = result.prices(l);
        out_shares[l] = result.shares(l);
    }
    *out_pp_ic = result.pp_ic;
    *out_outside = result.outside;
    *out_iters = result.iters;
}

EXPORT void eval_ic_c(const int* bundles_data, const double* prices_data, int L,
                      double* out_pp_ic, double* out_shares, double* out_outside) {
    VectorXi bundles = Map<const VectorXi>(bundles_data, L);
    VectorXd prices = Map<const VectorXd>(prices_data, L);

    MenuResult result = eval_ic(bundles, prices);

    for (int l = 0; l < L; ++l) {
        out_shares[l] = result.shares(l);
    }
    *out_pp_ic = result.pp_ic;
    *out_outside = result.outside;
}

// Diagnostic function to check if OpenMP is enabled
EXPORT int check_openmp_enabled() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 0;
#endif
}

}  // extern "C"

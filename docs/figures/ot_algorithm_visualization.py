"""
Create an intuitive visualization of the OT algorithm similar to K-means graphics.

This shows the iterative E-step (consumer assignment) and M-step (atom update)
in a 2D projection for pedagogical clarity.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green

np.random.seed(42)

# Generate synthetic consumer WTP data (2D for visualization)
# Three clusters representing different consumer segments
n_per_cluster = 30
cluster1 = np.random.multivariate_normal([15, 10], [[4, 0], [0, 2]], n_per_cluster)
cluster2 = np.random.multivariate_normal([35, 25], [[3, 0], [0, 3]], n_per_cluster)
cluster3 = np.random.multivariate_normal([50, 15], [[2, 0], [0, 4]], n_per_cluster)
consumers = np.vstack([cluster1, cluster2, cluster3])

# Initialize 3 menu items (randomly)
np.random.seed(123)
init_menus = consumers[np.random.choice(len(consumers), 3, replace=False)]

# Function to assign consumers to nearest menu item (E-step)
def assign_consumers(consumers, menus):
    """Assign each consumer to utility-maximizing menu item"""
    assignments = []
    for c in consumers:
        # In real algorithm: argmax(v·b - p)
        # Here we use Euclidean distance as proxy for visualization
        distances = np.linalg.norm(menus - c, axis=1)
        assignments.append(np.argmin(distances))
    return np.array(assignments)

# Function to update menu items (M-step)
def update_menus(consumers, assignments, n_menus):
    """Update each menu item to center of assigned consumers"""
    new_menus = np.zeros((n_menus, 2))
    for m in range(n_menus):
        assigned = consumers[assignments == m]
        if len(assigned) > 0:
            # In real algorithm: argmax profit over (b,p)
            # Here we use centroid as proxy
            new_menus[m] = assigned.mean(axis=0)
        else:
            new_menus[m] = menus[m]  # Keep if no assignments
    return new_menus

# Run algorithm for 3 iterations
menus_history = [init_menus.copy()]
assignments_history = []

menus = init_menus.copy()
for iteration in range(3):
    # E-step: Assign consumers
    assignments = assign_consumers(consumers, menus)
    assignments_history.append(assignments)

    # M-step: Update menus
    menus = update_menus(consumers, assignments, 3)
    menus_history.append(menus.copy())

# Create figure with 4 subplots (initialization + 3 iterations)
fig = plt.figure(figsize=(16, 4))

titles = [
    'Initialization\n(Random menu items)',
    'Iteration 1\nE-step: Assign consumers → M-step: Update atoms',
    'Iteration 2\nE-step: Assign consumers → M-step: Update atoms',
    'Iteration 3 (Converged)\nOptimal menu maximizes profit'
]

for idx in range(4):
    ax = fig.add_subplot(1, 4, idx + 1)

    if idx == 0:
        # Initial state: just show consumers and random menu items
        ax.scatter(consumers[:, 0], consumers[:, 1], c='gray', s=30,
                  alpha=0.4, edgecolors='k', linewidths=0.5, label='Consumers')
        ax.scatter(menus_history[0][:, 0], menus_history[0][:, 1],
                  c=colors, s=400, marker='*', edgecolors='k', linewidths=2,
                  label='Menu items', zorder=100)

        # Add labels for menu items
        for i, menu in enumerate(menus_history[0]):
            ax.annotate(f'Item {i+1}', xy=menu, xytext=(5, 5),
                       textcoords='offset points', fontsize=9, fontweight='bold')
    else:
        # Show assignments and updated menus
        iter_idx = idx - 1
        assignments = assignments_history[iter_idx]

        # Plot consumers colored by assignment
        for m in range(3):
            mask = assignments == m
            ax.scatter(consumers[mask, 0], consumers[mask, 1],
                      c=colors[m], s=30, alpha=0.6, edgecolors='k',
                      linewidths=0.5)

        # Plot old menu position (faded)
        ax.scatter(menus_history[iter_idx][:, 0], menus_history[iter_idx][:, 1],
                  c=colors, s=200, marker='*', alpha=0.3, edgecolors='gray',
                  linewidths=1, zorder=50)

        # Plot new menu position
        ax.scatter(menus_history[iter_idx + 1][:, 0], menus_history[iter_idx + 1][:, 1],
                  c=colors, s=400, marker='*', edgecolors='k', linewidths=2,
                  zorder=100)

        # Draw arrows showing movement
        for m in range(3):
            old_pos = menus_history[iter_idx][m]
            new_pos = menus_history[iter_idx + 1][m]
            if np.linalg.norm(old_pos - new_pos) > 0.5:
                arrow = FancyArrowPatch(old_pos, new_pos,
                                      arrowstyle='->', mutation_scale=20,
                                      lw=2, color=colors[m], alpha=0.7,
                                      zorder=90)
                ax.add_patch(arrow)

        # Add labels for menu items
        for i, menu in enumerate(menus_history[iter_idx + 1]):
            ax.annotate(f'Item {i+1}', xy=menu, xytext=(5, 5),
                       textcoords='offset points', fontsize=9, fontweight='bold')

    ax.set_xlim(5, 60)
    ax.set_ylim(0, 35)
    ax.set_xlabel('WTP for Feature 1', fontsize=10)
    ax.set_ylabel('WTP for Feature 2', fontsize=10)
    ax.set_title(titles[idx], fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

# Add legend to first subplot
fig.legend(['Consumers', 'Menu items (bundles + prices)'],
          loc='upper center', bbox_to_anchor=(0.5, 0.02),
          ncol=2, fontsize=10, frameon=True)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('d:/VSCode/curriculum_paper/optimaldiscretemenu-public/docs/figures/ot_algorithm_kmeans_style.png',
            dpi=300, bbox_inches='tight')
print("Saved: ot_algorithm_kmeans_style.png")

# Create second figure: Side-by-side comparison with actual joint (b,p) space
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# LEFT: WTP space (what we visualized above - final iteration)
ax1 = axes[0]
assignments = assignments_history[-1]
for m in range(3):
    mask = assignments == m
    ax1.scatter(consumers[mask, 0], consumers[mask, 1],
              c=colors[m], s=40, alpha=0.6, edgecolors='k',
              linewidths=0.5, label=f'Assigned to Item {m+1}')

ax1.scatter(menus_history[-1][:, 0], menus_history[-1][:, 1],
          c=colors, s=500, marker='*', edgecolors='k', linewidths=2,
          zorder=100)

for i, menu in enumerate(menus_history[-1]):
    ax1.annotate(f'Item {i+1}', xy=menu, xytext=(5, 5),
               textcoords='offset points', fontsize=10, fontweight='bold')

ax1.set_xlim(5, 60)
ax1.set_ylim(0, 35)
ax1.set_xlabel('WTP for Feature 1 (v₁)', fontsize=12)
ax1.set_ylabel('WTP for Feature 2 (v₂)', fontsize=12)
ax1.set_title('Consumer WTP Space\n(Euclidean projection for visualization)',
             fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=9)

# RIGHT: Joint (bundle, price) space - conceptual
ax2 = axes[1]

# Show discrete bundles (for d=2: {00, 01, 10, 11})
bundle_labels = ['∅', 'b₁', 'b₂', 'b₁+b₂']
bundle_x = [0, 1, 1, 2]  # number of features
bundle_colors_discrete = ['lightgray', '#ffb3ba', '#baffc9', '#bae1ff']

# Show 3 menu items as atoms in this space
# (x-axis: bundle size, y-axis: price)
menu_bundles = [0.5, 1.0, 1.8]  # bundle sizes (jittered for visualization)
menu_prices = [8, 18, 35]  # prices

ax2.scatter(menu_bundles, menu_prices, c=colors, s=600, marker='*',
           edgecolors='k', linewidths=2, zorder=100)

for i, (b, p) in enumerate(zip(menu_bundles, menu_prices)):
    ax2.annotate(f'Item {i+1}\n({bundle_labels[i+1]}, ${p:.0f})',
               xy=(b, p), xytext=(10, 10),
               textcoords='offset points', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))

    # Show Voronoi cells (profit regions)
    if i == 0:
        rect = FancyBboxPatch((0, 0), 0.75, 15,
                             boxstyle="round,pad=0.05",
                             edgecolor=colors[i], facecolor=colors[i],
                             alpha=0.15, linewidth=2, linestyle='--')
        ax2.add_patch(rect)
    elif i == 1:
        rect = FancyBboxPatch((0.75, 10), 0.8, 15,
                             boxstyle="round,pad=0.05",
                             edgecolor=colors[i], facecolor=colors[i],
                             alpha=0.15, linewidth=2, linestyle='--')
        ax2.add_patch(rect)
    else:
        rect = FancyBboxPatch((1.5, 25), 0.8, 20,
                             boxstyle="round,pad=0.05",
                             edgecolor=colors[i], facecolor=colors[i],
                             alpha=0.15, linewidth=2, linestyle='--')
        ax2.add_patch(rect)

ax2.set_xlim(-0.2, 2.5)
ax2.set_ylim(0, 50)
ax2.set_xlabel('Bundle size |b|', fontsize=12)
ax2.set_ylabel('Price p', fontsize=12)
ax2.set_title('Joint (Bundle, Price) Space\n(Actual optimization space)',
             fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add cost curve
bundle_range = np.linspace(0, 2.5, 100)
cost_curve = 5 * bundle_range + bundle_range**2
ax2.plot(bundle_range, cost_curve, 'k--', linewidth=2, alpha=0.5,
        label='Cost C(b) = 5|b| + |b|²')
ax2.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('d:/VSCode/curriculum_paper/optimaldiscretemenu-public/docs/figures/ot_wtp_vs_joint_space.png',
            dpi=300, bbox_inches='tight')
print("Saved: ot_wtp_vs_joint_space.png")

# Create third figure: Step-by-step breakdown
fig3 = plt.figure(figsize=(16, 10))

# Top row: E-step detail
ax_e1 = plt.subplot(2, 3, 1)
ax_e2 = plt.subplot(2, 3, 2)
ax_e3 = plt.subplot(2, 3, 3)

# Bottom row: M-step detail
ax_m1 = plt.subplot(2, 3, 4)
ax_m2 = plt.subplot(2, 3, 5)
ax_m3 = plt.subplot(2, 3, 6)

# E-STEP: Show one consumer choosing between menu items
iter_idx = 1
menus = menus_history[iter_idx]

# Pick one consumer from each cluster
sample_consumers = [consumers[15], consumers[45], consumers[75]]
sample_labels = ['Budget\nConsumer', 'Standard\nConsumer', 'Premium\nConsumer']

for subplot_idx, (ax, consumer, label) in enumerate(zip([ax_e1, ax_e2, ax_e3],
                                                          sample_consumers,
                                                          sample_labels)):
    # Plot all menu items
    ax.scatter(menus[:, 0], menus[:, 1], c=colors, s=300, marker='*',
              edgecolors='k', linewidths=2, alpha=0.5, zorder=50)

    # Highlight the consumer
    ax.scatter(consumer[0], consumer[1], c='black', s=200, marker='o',
              edgecolors='yellow', linewidths=3, zorder=100)

    # Draw lines to each menu item with utility values
    for m, menu in enumerate(menus):
        utility = -np.linalg.norm(consumer - menu)  # Proxy for v·b - p
        ax.plot([consumer[0], menu[0]], [consumer[1], menu[1]],
               'k--', alpha=0.3, linewidth=1)

        # Annotate with "utility"
        mid_x = (consumer[0] + menu[0]) / 2
        mid_y = (consumer[1] + menu[1]) / 2
        ax.text(mid_x, mid_y, f'u={utility:.1f}', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Find best menu item
    utilities = [-np.linalg.norm(consumer - menu) for menu in menus]
    best_m = np.argmax(utilities)

    # Highlight best choice
    ax.scatter(menus[best_m, 0], menus[best_m, 1], c=colors[best_m],
              s=500, marker='*', edgecolors='lime', linewidths=4, zorder=150)

    ax.annotate(f'CHOOSES\nItem {best_m+1}', xy=menus[best_m],
               xytext=(15, 15), textcoords='offset points',
               fontsize=10, fontweight='bold', color='green',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lime', alpha=0.3),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    ax.set_xlim(5, 60)
    ax.set_ylim(0, 35)
    ax.set_xlabel('WTP Feature 1', fontsize=9)
    ax.set_ylabel('WTP Feature 2', fontsize=9)
    ax.set_title(f'E-Step: {label}\nargmax(v·b - p)',
                fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)

# M-STEP: Show updating one menu item based on assigned consumers
for subplot_idx, (ax, m) in enumerate(zip([ax_m1, ax_m2, ax_m3], [0, 1, 2])):
    # Get consumers assigned to this menu item
    assignments = assignments_history[iter_idx]
    assigned = consumers[assignments == m]
    not_assigned = consumers[assignments != m]

    # Plot assigned consumers
    ax.scatter(assigned[:, 0], assigned[:, 1], c=colors[m], s=60,
              alpha=0.7, edgecolors='k', linewidths=0.5, label='Assigned')

    # Plot non-assigned (faded)
    ax.scatter(not_assigned[:, 0], not_assigned[:, 1], c='lightgray', s=20,
              alpha=0.3, edgecolors='none')

    # Old menu position
    old_menu = menus_history[iter_idx][m]
    ax.scatter(old_menu[0], old_menu[1], c=colors[m], s=300, marker='*',
              alpha=0.3, edgecolors='gray', linewidths=1, label='Old position')

    # New menu position (centroid of assigned consumers)
    new_menu = menus_history[iter_idx + 1][m]
    ax.scatter(new_menu[0], new_menu[1], c=colors[m], s=500, marker='*',
              edgecolors='k', linewidths=3, label='New position', zorder=100)

    # Arrow showing update
    arrow = FancyArrowPatch(old_menu, new_menu,
                          arrowstyle='->', mutation_scale=25,
                          lw=3, color=colors[m], alpha=0.8, zorder=90)
    ax.add_patch(arrow)

    # Show profit maximization concept
    if len(assigned) > 0:
        ax.annotate(f'Maximize profit\nfrom {len(assigned)} consumers',
                   xy=new_menu, xytext=(15, -25),
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[m], alpha=0.2),
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors[m]))

    ax.set_xlim(5, 60)
    ax.set_ylim(0, 35)
    ax.set_xlabel('WTP Feature 1', fontsize=9)
    ax.set_ylabel('WTP Feature 2', fontsize=9)
    ax.set_title(f'M-Step: Update Item {m+1}\nargmax profit over (b,p)',
                fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', fontsize=7)

plt.suptitle('Detailed Algorithm Steps: E-Step (Assignment) and M-Step (Update)',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('d:/VSCode/curriculum_paper/optimaldiscretemenu-public/docs/figures/ot_algorithm_detailed_steps.png',
            dpi=300, bbox_inches='tight')
print("Saved: ot_algorithm_detailed_steps.png")

plt.close('all')
print("\nGenerated 3 visualization files:")
print("1. ot_algorithm_kmeans_style.png - Main iterative visualization")
print("2. ot_wtp_vs_joint_space.png - WTP space vs. joint (b,p) space")
print("3. ot_algorithm_detailed_steps.png - Detailed E-step and M-step breakdown")

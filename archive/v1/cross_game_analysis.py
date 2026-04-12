"""
Cross-game analysis figures for Dream Perturbation experiment results.
Generates 5 publication-quality figures comparing Breakout and Pong.
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BREAKOUT_JSON = "/home/user/workspace/learning-from-failure/results_v2_breakout/dream_perturbation_results_v2.json"
PONG_JSON     = "/home/user/workspace/learning-from-failure/results_v2_pong/dream_perturbation_results_v2.json"
OUT_DIR       = "/home/user/workspace/learning-from-failure/results_v2_cross_game"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
with open(BREAKOUT_JSON) as f:
    breakout = json.load(f)
with open(PONG_JSON) as f:
    pong = json.load(f)

# ─── Constants ────────────────────────────────────────────────────────────────
AGENTS         = ["baseline", "single_perturb", "multi_dream", "adaptive_multi_dream"]
AGENT_LABELS   = ["Baseline", "Single-Perturb", "Multi-Dream", "Adaptive"]
VARIANTS       = ["normal", "mirrored", "noisy", "shifted_physics", "hard_mode"]
VARIANT_LABELS = ["Normal", "Mirrored", "Noisy", "Shifted\nPhysics", "Hard\nMode"]
COLORS         = {
    "baseline":           "#5B89C4",
    "single_perturb":     "#E8965D",
    "multi_dream":        "#72B77E",
    "adaptive_multi_dream": "#C44E52",
}
AGENT_COLOR_LIST = [COLORS[a] for a in AGENTS]
GAMES            = ["Breakout", "Pong"]
DATA             = {"Breakout": breakout, "Pong": pong}

plt.style.use('seaborn-v0_8-whitegrid')
TITLE_FS = 14
LABEL_FS = 12
TICK_FS  = 10
DPI      = 150


# ─── Helper: standard error ───────────────────────────────────────────────────
def std_err(data_dict, agent):
    std = data_dict["real_env"][agent]["std_return"]
    n   = data_dict["real_env"][agent]["num_episodes"]
    return std / math.sqrt(n)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: cross_game_real_env.png
# Side-by-side grouped bar chart, 4 agents per game, error bars = SE
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

n_agents  = len(AGENTS)
n_groups  = len(GAMES)
bar_width = 0.18
group_gap = 0.3
group_positions = np.array([0.0, 1.0 + group_gap])

for i, (agent, label, color) in enumerate(zip(AGENTS, AGENT_LABELS, AGENT_COLOR_LIST)):
    offsets = group_positions + (i - (n_agents - 1) / 2) * bar_width
    means   = [DATA[g]["real_env"][agent]["mean_return"] for g in GAMES]
    errors  = [std_err(DATA[g], agent) for g in GAMES]
    bars = ax.bar(offsets, means, bar_width, color=color, label=label,
                  yerr=errors, capsize=4, error_kw=dict(elinewidth=1.2, ecolor='black'))
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errors) * 0.1 + 0.5,
                f"{mean:.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(group_positions)
ax.set_xticklabels(GAMES, fontsize=TICK_FS)
ax.set_ylabel("Mean Episode Return", fontsize=LABEL_FS)
ax.set_title("Real Environment Performance Across Games", fontsize=TITLE_FS, fontweight='bold')
ax.legend(fontsize=TICK_FS, loc='upper right')
ax.tick_params(axis='y', labelsize=TICK_FS)
plt.tight_layout(pad=1.5)
out1 = os.path.join(OUT_DIR, "cross_game_real_env.png")
plt.savefig(out1, dpi=DPI)
plt.close()
print(f"Saved: {out1}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: cross_game_robustness.png
# Mean return averaged across all 5 dream variants
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for i, (agent, label, color) in enumerate(zip(AGENTS, AGENT_LABELS, AGENT_COLOR_LIST)):
    offsets = group_positions + (i - (n_agents - 1) / 2) * bar_width
    means   = []
    for g in GAMES:
        variant_means = [DATA[g]["dream_variants"][agent][v]["mean_return"] for v in VARIANTS]
        means.append(np.mean(variant_means))
    bars = ax.bar(offsets, means, bar_width, color=color, label=label)
    for bar, mean in zip(bars, means):
        offset = 0.003 if mean >= 0 else -0.015
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{mean:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(group_positions)
ax.set_xticklabels(GAMES, fontsize=TICK_FS)
ax.set_ylabel("Mean Return (avg. across dream variants)", fontsize=LABEL_FS)
ax.set_title("Dream Robustness Across Games", fontsize=TITLE_FS, fontweight='bold')
ax.legend(fontsize=TICK_FS, loc='upper right')
ax.tick_params(axis='y', labelsize=TICK_FS)
plt.tight_layout(pad=1.5)
out2 = os.path.join(OUT_DIR, "cross_game_robustness.png")
plt.savefig(out2, dpi=DPI)
plt.close()
print(f"Saved: {out2}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: cross_game_heatmap.png
# 8 rows (4 agents x 2 games) x 5 columns (dream variants)
# ──────────────────────────────────────────────────────────────────────────────
row_labels = []
heatmap_data = []

for game in GAMES:
    for agent, alabel in zip(AGENTS, AGENT_LABELS):
        row_labels.append(f"{game} - {alabel}")
        row = [DATA[game]["dream_variants"][agent][v]["mean_return"] for v in VARIANTS]
        heatmap_data.append(row)

heatmap_arr = np.array(heatmap_data)

# center colormap on 0
vmax = np.abs(heatmap_arr).max()
vmin = -vmax

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heatmap_arr, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')

# Annotate each cell
for r in range(heatmap_arr.shape[0]):
    for c in range(heatmap_arr.shape[1]):
        val = heatmap_arr[r, c]
        # choose text color based on background brightness
        bg_norm = (val - vmin) / (vmax - vmin)
        text_color = 'black' if 0.3 < bg_norm < 0.7 else 'white' if bg_norm < 0.3 else 'black'
        ax.text(c, r, f"{val:.2f}", ha='center', va='center',
                fontsize=9, fontweight='bold', color=text_color)

ax.set_xticks(range(len(VARIANTS)))
ax.set_xticklabels(["Normal", "Mirrored", "Noisy", "Shifted Physics", "Hard Mode"],
                   fontsize=TICK_FS)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=TICK_FS)
ax.set_title("Agent Robustness: Breakout vs Pong", fontsize=TITLE_FS, fontweight='bold')

# Add a dividing line between Breakout and Pong rows
ax.axhline(y=3.5, color='white', linewidth=2.5, linestyle='--')

cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Mean Return", fontsize=LABEL_FS)
cbar.ax.tick_params(labelsize=TICK_FS)

plt.tight_layout(pad=1.5)
out3 = os.path.join(OUT_DIR, "cross_game_heatmap.png")
plt.savefig(out3, dpi=DPI)
plt.close()
print(f"Saved: {out3}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: performance_robustness_tradeoff.png
# Scatter: X = normalized real env return, Y = mean dream robustness
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

markers = {'Breakout': 'o', 'Pong': 's'}
points  = []   # (x, y, agent_label, game, color)

for game in GAMES:
    raw_returns = [DATA[game]["real_env"][a]["mean_return"] for a in AGENTS]
    min_r, max_r = min(raw_returns), max(raw_returns)
    span = max_r - min_r if max_r != min_r else 1.0

    for agent, alabel, color in zip(AGENTS, AGENT_LABELS, AGENT_COLOR_LIST):
        x_raw   = DATA[game]["real_env"][agent]["mean_return"]
        x_norm  = (x_raw - min_r) / span
        y_rob   = np.mean([DATA[game]["dream_variants"][agent][v]["mean_return"] for v in VARIANTS])
        points.append((x_norm, y_rob, alabel, game, color))

# Plot points
for x, y, alabel, game, color in points:
    ax.scatter(x, y, marker=markers[game], color=color, s=120, zorder=5,
               edgecolors='black', linewidths=0.7)
    # label offset
    ax.annotate(alabel, (x, y), textcoords="offset points", xytext=(6, 5),
                fontsize=8.5, fontweight='bold', color=color)

# Pareto frontier (maximize both x and y simultaneously)
# A point is Pareto-dominant if no other point is better in both dimensions
pts_arr = np.array([(p[0], p[1]) for p in points])
n_pts = len(pts_arr)
pareto_mask = np.ones(n_pts, dtype=bool)
for i in range(n_pts):
    for j in range(n_pts):
        if i != j:
            # j dominates i if j >= i on both and strictly better on at least one
            if pts_arr[j, 0] >= pts_arr[i, 0] and pts_arr[j, 1] >= pts_arr[i, 1] and \
               (pts_arr[j, 0] > pts_arr[i, 0] or pts_arr[j, 1] > pts_arr[i, 1]):
                pareto_mask[i] = False
                break

pareto_pts = pts_arr[pareto_mask]
if len(pareto_pts) > 1:
    # Sort by x for the frontier line
    sorted_pareto = pareto_pts[np.argsort(pareto_pts[:, 0])]
    # Draw a staircase Pareto frontier
    px, py = sorted_pareto[:, 0], sorted_pareto[:, 1]
    step_x = [px[0]]
    step_y = [py[0]]
    for k in range(1, len(px)):
        step_x.append(px[k])
        step_y.append(step_y[-1])
        step_x.append(px[k])
        step_y.append(py[k])
    ax.plot(step_x, step_y, 'k--', linewidth=1.5, alpha=0.6, label='Pareto Frontier')

# Legend for games and agents
game_handles = [
    Line2D([0], [0], marker='o', color='gray', markersize=9, linestyle='None', label='Breakout'),
    Line2D([0], [0], marker='s', color='gray', markersize=9, linestyle='None', label='Pong'),
]
agent_handles = [
    mpatches.Patch(color=COLORS[a], label=l)
    for a, l in zip(AGENTS, AGENT_LABELS)
]
if len(pareto_pts) > 1:
    pareto_handle = [Line2D([0], [0], linestyle='--', color='black', label='Pareto Frontier')]
else:
    pareto_handle = []

first_legend  = ax.legend(handles=game_handles, title='Game', fontsize=TICK_FS,
                           loc='upper left', bbox_to_anchor=(0.0, 1.0))
ax.add_artist(first_legend)
second_legend = ax.legend(handles=agent_handles + pareto_handle, title='Agent',
                           fontsize=TICK_FS, loc='upper left', bbox_to_anchor=(0.18, 1.0))

ax.set_xlabel("Normalized Real Env Return (per game, 0–1)", fontsize=LABEL_FS)
ax.set_ylabel("Mean Dream Robustness (avg. across variants)", fontsize=LABEL_FS)
ax.set_title("Performance vs Robustness Tradeoff", fontsize=TITLE_FS, fontweight='bold')
ax.tick_params(labelsize=TICK_FS)
plt.tight_layout(pad=1.5)
out4 = os.path.join(OUT_DIR, "performance_robustness_tradeoff.png")
plt.savefig(out4, dpi=DPI)
plt.close()
print(f"Saved: {out4}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: adaptive_curriculum_focus.png
# 2x1 subplot: adaptive minus baseline per dream variant, per game
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, game in zip(axes, GAMES):
    diffs = []
    for v in VARIANTS:
        adaptive_val = DATA[game]["dream_variants"]["adaptive_multi_dream"][v]["mean_return"]
        baseline_val = DATA[game]["dream_variants"]["baseline"][v]["mean_return"]
        diffs.append(adaptive_val - baseline_val)

    bar_colors = ['#2ca02c' if d >= 0 else '#d62728' for d in diffs]
    x = np.arange(len(VARIANTS))
    bars = ax.bar(x, diffs, color=bar_colors, edgecolor='black', linewidth=0.6)

    ax.axhline(0, color='black', linewidth=1.0)
    for bar, diff in zip(bars, diffs):
        va    = 'bottom' if diff >= 0 else 'top'
        ypos  = bar.get_height() if diff >= 0 else bar.get_height()
        offset = 0.005 if diff >= 0 else -0.005
        ax.text(bar.get_x() + bar.get_width() / 2, ypos + offset,
                f"{diff:+.3f}", ha='center', va=va, fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(["Normal", "Mirrored", "Noisy", "Shifted\nPhysics", "Hard\nMode"],
                       fontsize=TICK_FS)
    ax.set_ylabel("Return Difference (Adaptive − Baseline)", fontsize=LABEL_FS)
    ax.set_title(game, fontsize=TITLE_FS, fontweight='bold')
    ax.tick_params(axis='y', labelsize=TICK_FS)

# Add legend patches
green_patch = mpatches.Patch(color='#2ca02c', label='Adaptive better')
red_patch   = mpatches.Patch(color='#d62728', label='Baseline better')
fig.legend(handles=[green_patch, red_patch], loc='upper center',
           ncol=2, fontsize=TICK_FS, bbox_to_anchor=(0.5, 1.02))

fig.suptitle("Adaptive Curriculum: Improvement Over Baseline by Dream Variant",
             fontsize=TITLE_FS, fontweight='bold', y=1.06)
plt.tight_layout(pad=1.5)
out5 = os.path.join(OUT_DIR, "adaptive_curriculum_focus.png")
plt.savefig(out5, dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Saved: {out5}")


# ─── Final verification ───────────────────────────────────────────────────────
print("\nVerification:")
for path in [out1, out2, out3, out4, out5]:
    size = os.path.getsize(path)
    status = "OK" if size > 0 else "EMPTY!"
    print(f"  {os.path.basename(path)}: {size:,} bytes [{status}]")

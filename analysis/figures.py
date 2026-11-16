import sys
# Phase 2 data workbook ("Data Entry" sheet). Not published: it holds participant responses.
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/phase2_data.xlsx"
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load data ─────────────────────────────────────────────────
df = pd.read_excel(
    DATA,
    sheet_name="Data Entry", header=2
)
df = df[pd.to_numeric(df['#'], errors='coerce').notna()].reset_index(drop=True)

ecl_s  = pd.to_numeric(df.iloc[:, 14], errors='coerce').dropna().values
ecl_m  = pd.to_numeric(df.iloc[:, 15], errors='coerce').dropna().values
gs_s   = pd.to_numeric(df.iloc[:, 28], errors='coerce').dropna().values
gs_m   = pd.to_numeric(df.iloc[:, 29], errors='coerce').dropna().values
ros_s  = pd.to_numeric(df.iloc[:, 42], errors='coerce').dropna().values
ros_m  = pd.to_numeric(df.iloc[:, 43], errors='coerce').dropna().values
profile = df['Profile'].values

n = len(gs_s)

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
})

COLOR_S = '#2563EB'   # blue  — Spontaneous
COLOR_M = '#9CA3AF'   # gray  — Mechanical
COLOR_PB = '#EA580C'  # orange — Pattern Breaker
COLOR_WC = '#2563EB'  # blue  — Watchful Companion
COLOR_AP = '#6B7280'  # gray  — Ambient Presence

# ── Figure 1: Bar chart H1 Animacy ───────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4))

means = [gs_s.mean(), gs_m.mean()]
sds   = [gs_s.std(ddof=1), gs_m.std(ddof=1)]
ses   = [s / np.sqrt(n) for s in sds]
bars  = ax.bar(['Spontaneous\n(VLM)', 'Mechanical\n(Control)'],
               means, yerr=ses, capsize=5,
               color=[COLOR_S, COLOR_M], width=0.5,
               error_kw={'linewidth': 1.2, 'ecolor': '#374151'})

ax.set_ylabel('Godspeed Animacy (1–7)', fontsize=11)
ax.set_ylim(0, 7.5)
ax.set_title('H1 — Godspeed Animacy Score', fontsize=12, fontweight='bold', pad=10)

# Annotate significance
y_line = max(means) + max(ses) + 0.3
ax.plot([0, 1], [y_line, y_line], color='black', linewidth=0.8)
ax.text(0.5, y_line + 0.1, 'p = .009 **', ha='center', fontsize=10)

# Mean labels inside bars
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, m/2,
            f'{m:.2f}', ha='center', va='center',
            color='white', fontsize=11, fontweight='bold')

ax.text(0.98, 0.05, "Cohen's d = 0.73\n(large effect)",
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9, color='#374151')

plt.tight_layout()
plt.savefig('H1_animacy_bar.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: H1_animacy_bar.png")

# ── Figure 2: Paired line chart H1 ───────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4.5))

profile_colors = {'PB': COLOR_PB, 'WC': COLOR_WC, 'AP': COLOR_AP}

for i in range(n):
    p = profile[i]
    c = profile_colors.get(p, '#9CA3AF')
    ax.plot([0, 1], [gs_s[i], gs_m[i]],
            color=c, alpha=0.6, linewidth=1.2, zorder=2)
    ax.scatter([0, 1], [gs_s[i], gs_m[i]],
               color=c, s=30, zorder=3, alpha=0.8)

# Group means
ax.plot([0, 1], [gs_s.mean(), gs_m.mean()],
        color='black', linewidth=2.5, zorder=4, label='Group mean')
ax.scatter([0, 1], [gs_s.mean(), gs_m.mean()],
           color='black', s=60, zorder=5)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Spontaneous\n(VLM)', 'Mechanical\n(Control)'])
ax.set_ylabel('Godspeed Animacy (1–7)', fontsize=11)
ax.set_ylim(0.5, 7.5)
ax.set_title('H1 — Individual Paired Scores', fontsize=12, fontweight='bold', pad=10)

legend_handles = [
    mpatches.Patch(color=COLOR_PB, label='Pattern Breaker'),
    mpatches.Patch(color=COLOR_WC, label='Watchful Companion'),
    mpatches.Patch(color=COLOR_AP, label='Ambient Presence'),
    mpatches.Patch(color='black',  label='Group mean'),
]
ax.legend(handles=legend_handles, fontsize=8,
          loc='lower right', framealpha=0.9)

ax.text(0.5, 0.04,
        f't(16) = 2.99,  p = .009,  r = {np.sqrt(2.992**2/(2.992**2+16)):.2f}',
        transform=ax.transAxes, ha='center', fontsize=8.5, color='#374151')

plt.tight_layout()
plt.savefig('H1_animacy_paired.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: H1_animacy_paired.png")

# ── Figure 3: Forest plot H2 non-inferiority ─────────────────
fig, ax = plt.subplots(figsize=(6, 2.8))

labels = ['Klepsch ECL\n(H2a)', 'RoSAS Discomfort\n(H2b)']
diffs  = [
    (ecl_s - ecl_m).mean(),
    (ros_s - ros_m).mean(),
]
uppers = [0.115, -0.154]   # pre-computed one-sided 95% CI upper
lowers_display = [
    diffs[0] - (uppers[0] - diffs[0]),
    diffs[1] - (uppers[1] - diffs[1]),
]

colors = ['#1D9E75', '#1D9E75']
y = [1, 0]

for i in range(2):
    ax.plot([lowers_display[i], uppers[i]], [y[i], y[i]],
            color=colors[i], linewidth=2.5, solid_capstyle='round')
    ax.scatter(diffs[i], y[i], color=colors[i], s=60, zorder=5)
    ax.text(uppers[i] + 0.03, y[i],
            f'Δ = {diffs[i]:.2f},  upper CI = {uppers[i]:.3f}',
            va='center', fontsize=9, color='#374151')

# Non-inferiority boundary
ax.axvline(x=0.5, color='#E24B4A', linewidth=1.2,
           linestyle='--', label='Non-inferiority boundary (Δ=0.5)')
ax.axvline(x=0, color='#374151', linewidth=0.6, linestyle=':')

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Mean Difference (Spontaneous − Mechanical)', fontsize=10)
ax.set_xlim(-2, 1.5)
ax.set_ylim(-0.5, 1.8)
ax.set_title('H2 — Non-inferiority Tests', fontsize=12, fontweight='bold', pad=10)
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.9)

ax.text(0.52, 1.72, 'boundary', color='#E24B4A',
        fontsize=8, transform=ax.get_xaxis_transform())

plt.tight_layout()
plt.savefig('H2_noninferiority.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: H2_noninferiority.png")

print("\nAll 3 figures saved in the same folder as this script.")

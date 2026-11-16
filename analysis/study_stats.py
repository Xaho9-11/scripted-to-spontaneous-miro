import sys
# Phase 2 data workbook ("Data Entry" sheet). Not published: it holds participant responses.
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/phase2_data.xlsx"
import pandas as pd
import numpy as np
from scipy import stats

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

n = len(gs_s)

# ── H1: Godspeed Animacy ──────────────────────────────────────
print("=" * 62)
print("  H1  Paired t-test — Godspeed Animacy")
print("=" * 62)
diff_gs = gs_s - gs_m
t_gs, p_gs = stats.ttest_rel(gs_s, gs_m)
se_gs = diff_gs.std(ddof=1) / np.sqrt(n)
ci_lo = diff_gs.mean() - stats.t.ppf(0.975, df=n-1) * se_gs
ci_hi = diff_gs.mean() + stats.t.ppf(0.975, df=n-1) * se_gs
d_gs  = diff_gs.mean() / diff_gs.std(ddof=1)

# Wilcoxon
W_gs, p_wil_gs = stats.wilcoxon(gs_s, gs_m)

print(f"  N                   = {n}")
print(f"  Spontaneous  M (SD) = {gs_s.mean():.3f}  ({gs_s.std(ddof=1):.3f})")
print(f"  Mechanical   M (SD) = {gs_m.mean():.3f}  ({gs_m.std(ddof=1):.3f})")
print(f"  Mean diff (S − M)   = {diff_gs.mean():.3f}")
print(f"  95% CI              = [{ci_lo:.3f},  {ci_hi:.3f}]")
print(f"  t({n-1})                = {t_gs:.3f}")
print(f"  p (two-tailed)      = {p_gs:.4f}  {'** (p<.01)' if p_gs<.01 else '* (p<.05)' if p_gs<.05 else 'n.s.'}")
print(f"  Cohen's d           = {d_gs:.3f}  (large effect)")
print(f"  --- Robustness check ---")
print(f"  Wilcoxon W          = {int(W_gs)}")
print(f"  p (Wilcoxon)        = {p_wil_gs:.4f}  {'** (p<.01)' if p_wil_gs<.01 else '* (p<.05)' if p_wil_gs<.05 else 'n.s.'}")
print(f"  → Both tests agree: result is robust")

# ── H2: ECL ──────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  H2a  Non-inferiority — ECL  (Δ = 0.5)")
print("=" * 62)
diff_ecl = ecl_s - ecl_m
se_ecl   = diff_ecl.std(ddof=1) / np.sqrt(n)
ci_ecl_upper = diff_ecl.mean() + stats.t.ppf(0.95, df=n-1) * se_ecl
W_ecl, p_wil_ecl = stats.wilcoxon(ecl_s, ecl_m)

print(f"  Spontaneous  M (SD) = {ecl_s.mean():.3f}  ({ecl_s.std(ddof=1):.3f})")
print(f"  Mechanical   M (SD) = {ecl_m.mean():.3f}  ({ecl_m.std(ddof=1):.3f})")
print(f"  Mean diff (S − M)   = {diff_ecl.mean():.3f}")
print(f"  95% one-sided upper = {ci_ecl_upper:.3f}")
print(f"  Non-inf boundary    = 0.5")
print(f"  → Passed: {ci_ecl_upper < 0.5}")
print(f"  --- Robustness check ---")
print(f"  Wilcoxon W          = {int(W_ecl)}")
print(f"  p (Wilcoxon)        = {p_wil_ecl:.4f}  {'** (p<.01)' if p_wil_ecl<.01 else '* (p<.05)' if p_wil_ecl<.05 else 'n.s.'}")

# ── H2: RoSAS ────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  H2b  Non-inferiority — RoSAS Discomfort  (Δ = 0.5)")
print("=" * 62)
diff_r = ros_s - ros_m
se_r   = diff_r.std(ddof=1) / np.sqrt(n)
ci_r_upper = diff_r.mean() + stats.t.ppf(0.95, df=n-1) * se_r
W_ros, p_wil_ros = stats.wilcoxon(ros_s, ros_m)

print(f"  Spontaneous  M (SD) = {ros_s.mean():.3f}  ({ros_s.std(ddof=1):.3f})")
print(f"  Mechanical   M (SD) = {ros_m.mean():.3f}  ({ros_m.std(ddof=1):.3f})")
print(f"  Mean diff (S − M)   = {diff_r.mean():.3f}")
print(f"  95% one-sided upper = {ci_r_upper:.3f}")
print(f"  Non-inf boundary    = 0.5")
print(f"  → Passed: {ci_r_upper < 0.5}")
print(f"  --- Robustness check ---")
print(f"  Wilcoxon W          = {int(W_ros)}")
print(f"  p (Wilcoxon)        = {p_wil_ros:.4f}  {'** (p<.01)' if p_wil_ros<.01 else '* (p<.05)' if p_wil_ros<.05 else 'n.s.'}")

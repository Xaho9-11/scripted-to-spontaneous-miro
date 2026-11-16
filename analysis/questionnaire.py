#!/usr/bin/env python3
"""
M12 Phase 2 — Questionnaire Analysis
Usage: questionnaire.py [phase2_data.xlsx]
Outputs results_summary.txt.
"""

import openpyxl, math, statistics, sys
from pathlib import Path

XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "phase2_data.xlsx"  # not published

# ── load data ──────────────────────────────────────────────────────────────────
wb = openpyxl.load_workbook(XLSX, data_only=True)
ws = wb["Data Entry"]

def col_vals(col_idx, rows=range(5,21)):
    """Return non-None numeric values from a column (1-indexed)."""
    return [ws.cell(r, col_idx).value for r in rows
            if ws.cell(r, col_idx).value is not None]

# mean columns (1-indexed, row 22 = formula)
# ECL S=K(11) M=L(12), GS S=Y(25) M=Z(26), RoSAS S=AM(39) M=AN(40)
# Habituation raw items: Habit_S1..S4 = AO..AR (41..44), Habit_M1..M4 = AS..AV (45..48)

def safe_mean(lst): return statistics.mean(lst) if lst else float("nan")
def safe_sd(lst):   return statistics.stdev(lst) if len(lst)>1 else float("nan")

ecl_s  = col_vals(11)
ecl_m  = col_vals(12)
gs_s   = col_vals(25)
gs_m   = col_vals(26)
ros_s  = col_vals(39)
ros_m  = col_vals(40)

hab_s  = [col_vals(c) for c in range(41,45)]  # 4 items
hab_m  = [col_vals(c) for c in range(45,49)]

# ── paired t-test ──────────────────────────────────────────────────────────────
def paired_t(a, b):
    """Returns (t, df, p, cohen_d, mean_diff, ci95_lo, ci95_hi)."""
    pairs = [(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    if len(pairs) < 2:
        return (float("nan"),)*7
    diffs = [x-y for x,y in pairs]
    n     = len(diffs)
    md    = statistics.mean(diffs)
    sd_d  = statistics.stdev(diffs)
    se    = sd_d / math.sqrt(n)
    t     = md / se
    df    = n - 1
    d     = md / sd_d   # Cohen's d
    # 95% CI (approximate using t-critical ≈ 2.145 for df=14)
    # More precise: use scipy if available, else approximate
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
        t_crit = stats.t.ppf(0.975, df)
    except ImportError:
        p = float("nan")
        t_crit = 2.145   # df=14 approx
    ci_lo = md - t_crit * se
    ci_hi = md + t_crit * se
    return t, df, p, d, md, ci_lo, ci_hi

# ── non-inferiority check (one-sided, boundary delta) ─────────────────────────
def non_inferiority(a, b, delta=0.5):
    """H2: is upper bound of 95% CI for (S-M) below delta?"""
    t, df, p, d, md, ci_lo, ci_hi = paired_t(a, b)
    result = "PASS ✓" if ci_hi < delta else "FAIL ✗ (Spontaneous may be worse)"
    return ci_hi, result

# ── format helpers ─────────────────────────────────────────────────────────────
def fmt(v, dec=2): return f"{v:.{dec}f}" if not math.isnan(v) else "N/A"
def pval(p):
    if math.isnan(p): return "N/A"
    if p < 0.001: return "< .001"
    if p < 0.01:  return f"= .{int(p*1000):03d}"[:-1]
    return f"= {p:.3f}"

# ── write report ───────────────────────────────────────────────────────────────
lines = []
N = len(ecl_s)
lines.append("=" * 65)
lines.append(f"  M12 Phase 2 — Questionnaire Analysis    N = {N}")
lines.append("=" * 65)

def report_scale(name, s_vals, m_vals, hypothesis, delta=None):
    n = len([x for x,y in zip(s_vals,m_vals) if x and y])
    t,df,p,d,md,ci_lo,ci_hi = paired_t(s_vals, m_vals)
    lines.append(f"\n── {name}  (n={n})")
    lines.append(f"   Spontaneous : M = {fmt(safe_mean(s_vals))}  SD = {fmt(safe_sd(s_vals))}")
    lines.append(f"   Mechanical  : M = {fmt(safe_mean(m_vals))}  SD = {fmt(safe_sd(m_vals))}")
    lines.append(f"   Difference (S−M) : {fmt(md)}  95% CI [{fmt(ci_lo)}, {fmt(ci_hi)}]")
    lines.append(f"   Paired t({int(df) if not math.isnan(df) else 'N/A'}) = {fmt(t)}  p {pval(p)}  Cohen's d = {fmt(d)}")
    if hypothesis == "H1":
        sig = "✓ H1 SUPPORTED" if (not math.isnan(p) and p < 0.05 and md > 0) else "✗ H1 NOT supported"
        lines.append(f"   → {sig}")
    elif hypothesis == "H2" and delta is not None:
        ci_up, result = non_inferiority(s_vals, m_vals, delta)
        lines.append(f"   Non-inferiority boundary Δ = {delta}")
        lines.append(f"   Upper CI bound = {fmt(ci_up)} → {result}")

report_scale("Godspeed Animacy  [H1 PRIMARY]",  gs_s,  gs_m,  "H1")
report_scale("RoSAS Discomfort  [H2 constraint]",ros_s, ros_m, "H2", delta=0.5)
report_scale("Klepsch ECL       [H2 constraint]",ecl_s, ecl_m, "H2", delta=0.5)

lines.append("\n── Habituation / Within-Session (Exploratory)")
hab_labels = ["Less Attention", "Predictable", "Fresh/Natural (R-coded)", "Matched Vibe"]
for i in range(4):
    sv, mv = hab_s[i], hab_m[i]
    if not sv or not mv: continue
    t,df,p,d,md,ci_lo,ci_hi = paired_t(sv,mv)
    lines.append(f"   {hab_labels[i]:28s}  S={fmt(safe_mean(sv))}  M={fmt(safe_mean(mv))}"
                 f"  diff={fmt(md)}  p {pval(p)}")

lines.append("\n" + "=" * 65)
lines.append("  Manipulation Check")
lines.append("  GS Animacy: S > M ?  " +
             ("YES ✓" if safe_mean(gs_s) > safe_mean(gs_m) else "NO ✗ — manipulation may have failed"))
lines.append("=" * 65)

out = "\n".join(lines)
print(out)
(Path(__file__).resolve().parent / "results_summary.txt").write_text(out)
print("\nSaved → results_summary.txt")

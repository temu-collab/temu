# -*- coding: utf-8 -*-
"""Convergence timing, phase-restricted trends and sample-efficiency statistics."""
import numpy as np, pandas as pd, os
from scipy import stats

F1 = r"C:\Users\hp\Desktop\approach1_insar_only progress.csv"
F2 = r"C:\Users\hp\Desktop\approach2_with_rainfall progress.csv"
OUT = r"C:\Users\hp\Desktop\training_progress_analysis"

d1 = pd.read_csv(F1).dropna(subset=['train/loss']).reset_index(drop=True)
d2 = pd.read_csv(F2).dropna(subset=['train/loss']).reset_index(drop=True)
ts = d1['time/total_timesteps'].values.astype(float)
it = d1['time/iterations'].values.astype(int)
METRICS = {'train/loss': 'Total loss', 'train/value_loss': 'Value loss',
           'train/entropy_loss': 'Entropy loss', 'train/explained_variance': 'Explained variance'}
pi = np.arange(32, 48)   # last third = iterations 34-49

print("=" * 100)
print("A. CONVERGENCE TIME (5-iteration moving average first enters and stays in +/-10% band of plateau)")
print("=" * 100)
rows = []
for col, lab in METRICS.items():
    for tag, d in [('A1', d1), ('A2', d2)]:
        x = d[col].values.astype(float)
        sm = pd.Series(x).rolling(5, center=True, min_periods=1).mean().values
        plateau = x[pi].mean()
        inside = np.abs(sm - plateau) <= 0.10 * abs(plateau)
        k = next((i for i in range(len(sm)) if inside[i:].all()), np.nan)
        # also 90% of total improvement achieved
        tot_impr = x[0] - plateau
        k90 = next((i for i in range(len(sm)) if (x[0] - sm[i]) >= 0.90 * tot_impr), np.nan)
        rows.append(dict(Metric=lab, Approach=tag, Plateau=plateau,
                         Conv_it=it[int(k)] if k == k else np.nan,
                         Conv_steps=ts[int(k)] if k == k else np.nan,
                         t90_it=it[int(k90)] if k90 == k90 else np.nan,
                         t90_steps=ts[int(k90)] if k90 == k90 else np.nan))
conv = pd.DataFrame(rows)
print(conv.to_string(index=False, float_format=lambda v: f"{v:10.1f}"))
conv.to_csv(os.path.join(OUT, "table_convergence_time.csv"), index=False)

print("\n" + "=" * 100)
print("B. TREND AFTER THE INITIAL TRANSIENT (iterations 12-49, i.e. >= 22,528 steps)")
print("=" * 100)
sub = np.arange(10, 48)
for col, lab in METRICS.items():
    for tag, d in [('A1', d1), ('A2', d2)]:
        x = d[col].values[sub]
        rho, p = stats.spearmanr(ts[sub], x)
        lr = stats.linregress(ts[sub], x)
        print(f"{lab:20s} {tag}: Spearman rho={rho:+.3f} (p={p:.2e}) | OLS slope={lr.slope*1e4:+9.4f} "
              f"per 10k steps (95% CI {(lr.slope-1.96*lr.stderr)*1e4:+.4f},{(lr.slope+1.96*lr.stderr)*1e4:+.4f}) "
              f"R2={lr.rvalue**2:.3f} p={lr.pvalue:.2e}")

print("\n" + "=" * 100)
print("C. LATE-PHASE (last 16 iterations) LOCAL TREND")
print("=" * 100)
for col, lab in METRICS.items():
    for tag, d in [('A1', d1), ('A2', d2)]:
        x = d[col].values[pi]
        lr = stats.linregress(ts[pi], x)
        print(f"{lab:20s} {tag}: slope={lr.slope*1e4:+9.4f} per 10k steps, p={lr.pvalue:.3f}, R2={lr.rvalue**2:.3f}")

print("\n" + "=" * 100)
print("D. SAMPLE EFFICIENCY: when does each run first reach A1's final plateau level?")
print("=" * 100)
for col, lab in [('train/loss', 'Total loss'), ('train/value_loss', 'Value loss')]:
    target = d1[col].values[pi].mean()
    for tag, d in [('A1', d1), ('A2', d2)]:
        sm = pd.Series(d[col].values).rolling(5, center=True, min_periods=1).mean().values
        k = next((i for i in range(len(sm)) if (sm[i:] <= target).all()), np.nan)
        print(f"{lab:12s} target={target:8.2f} (A1 plateau) -> {tag} reaches and holds it at "
              f"{'it %d / %d steps' % (it[int(k)], ts[int(k)]) if k == k else 'never'}")
    # A2 advantage in steps to reach A1 plateau
print("\n" + "=" * 100)
print("E. DISTRIBUTIONAL OVERLAP OF PLATEAU SAMPLES (A1 vs A2)")
print("=" * 100)
for col, lab in METRICS.items():
    x, y = d1[col].values[pi], d2[col].values[pi]
    ks = stats.ks_2samp(x, y)
    # Hedges g
    sp = np.sqrt(((len(x)-1)*x.var(ddof=1) + (len(y)-1)*y.var(ddof=1)) / (len(x)+len(y)-2))
    g = (x.mean() - y.mean()) / sp * (1 - 3/(4*(len(x)+len(y))-9))
    print(f"{lab:20s}: KS D={ks.statistic:.3f} p={ks.pvalue:.2e} | Hedges g={g:+.2f} | "
          f"range A1 [{x.min():.3f},{x.max():.3f}] vs A2 [{y.min():.3f},{y.max():.3f}] | "
          f"overlap={'YES' if (x.min() <= y.max() and y.min() <= x.max()) else 'NONE'}")

print("\n" + "=" * 100)
print("F. VALUE-LOSS / ENTROPY COUPLING AND WITHIN-RUN NOISE")
print("=" * 100)
for tag, d in [('A1', d1), ('A2', d2)]:
    vl = d['train/value_loss'].values
    resid = vl[pi] - pd.Series(vl).rolling(5, center=True, min_periods=1).mean().values[pi]
    print(f"{tag}: plateau residual SD (around 5-it MA) = {resid.std(ddof=1):.2f} "
          f"({100*resid.std(ddof=1)/vl[pi].mean():.1f}% of plateau mean) | "
          f"lag-1 autocorr of plateau value loss = {pd.Series(vl[pi]).autocorr(1):+.3f}")

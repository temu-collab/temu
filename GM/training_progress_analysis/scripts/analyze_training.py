# -*- coding: utf-8 -*-
"""
Statistical analysis of PPO training-progress logs
Approach 1 (InSAR only) vs Approach 2 (InSAR + rainfall)
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import json, os

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)

F1 = r"C:\Users\hp\Desktop\approach1_insar_only progress.csv"
F2 = r"C:\Users\hp\Desktop\approach2_with_rainfall progress.csv"
V7 = r"C:\Users\hp\Desktop\NR7\GM\rl_outputs_v7"
OUT = r"C:\Users\hp\Desktop\training_progress_analysis"
os.makedirs(OUT, exist_ok=True)

a1 = pd.read_csv(F1)
a2 = pd.read_csv(F2)

print("=" * 100)
print("1. DATA INTEGRITY / COMPLETENESS")
print("=" * 100)
for nm, d in [("A1 InSAR-only", a1), ("A2 +rainfall", a2)]:
    print(f"\n{nm}: shape={d.shape}")
    print("  columns:", list(d.columns))
    print("  timesteps:", d['time/total_timesteps'].min(), "->", d['time/total_timesteps'].max(),
          "| step size unique:", np.unique(np.diff(d['time/total_timesteps'])))
    print("  iterations:", d['time/iterations'].min(), "->", d['time/iterations'].max(),
          "| monotonic:", d['time/iterations'].is_monotonic_increasing)
    na = d.isna().sum()
    print("  rows with any NaN:", d.isna().any(axis=1).sum(), "| NaN cols:", dict(na[na > 0]))
    print("  duplicate rows:", d.duplicated().sum())
    print("  wall-clock (s):", d['time/time_elapsed'].max(), "| mean fps:", round(d['time/fps'].mean(), 1))
    print("  const hyperparams: clip_range=", d['train/clip_range'].dropna().unique(),
          " lr=", d['train/learning_rate'].dropna().unique())

# identical environment schedule check
same_len = np.allclose(a1['rollout/ep_len_mean'], a2['rollout/ep_len_mean'])
print("\n  ep_len_mean identical across approaches:", same_len,
      "  (max abs diff = %.3g)" % np.abs(a1['rollout/ep_len_mean'] - a2['rollout/ep_len_mean']).max())
print("  n_updates per iteration:", np.unique(np.diff(a1['train/n_updates'].dropna())))

# provenance check vs v7 folder logs
print("\n  -- provenance check vs rl_outputs_v7 sb3 logs --")
for tag, sub, ref in [("A1", "approach1_insar_only", a1), ("A2", "approach2_with_rainfall", a2)]:
    p = os.path.join(V7, sub, "sb3_log", "progress.csv")
    if os.path.exists(p):
        v = pd.read_csv(p)
        ident = v.shape == ref.shape and np.allclose(
            v['train/value_loss'].values[1:], ref['train/value_loss'].values[1:], rtol=1e-9)
        print(f"   {tag}: v7-folder final value_loss={v['train/value_loss'].iloc[-1]:.2f} "
              f"EV={v['train/explained_variance'].iloc[-1]:.3f} | supplied final "
              f"value_loss={ref['train/value_loss'].iloc[-1]:.2f} EV={ref['train/explained_variance'].iloc[-1]:.3f}"
              f" | byte-identical run: {ident}")

# ---------------------------------------------------------------- metrics
METRICS = {
    'train/loss':          'Total loss',
    'train/value_loss':    'Value loss',
    'train/entropy_loss':  'Entropy loss',
}
SUPPORT = {
    'train/explained_variance': 'Explained variance',
    'train/policy_gradient_loss': 'Policy-gradient loss',
    'train/approx_kl': 'Approx. KL',
    'train/clip_fraction': 'Clip fraction',
    'rollout/ep_rew_mean': 'Episode reward',
}

d1 = a1.dropna(subset=['train/loss']).reset_index(drop=True)
d2 = a2.dropna(subset=['train/loss']).reset_index(drop=True)
n = len(d1)
print(f"\n  logged training iterations (excluding bootstrap row): n={n} per approach")

# phases: thirds of the logged iterations
idx = np.arange(n)
ph = {'early (it 2-17)': idx[:16], 'mid (it 18-33)': idx[16:32], 'late (it 34-49)': idx[32:]}

print("\n" + "=" * 100)
print("2. DESCRIPTIVE STATISTICS (all logged iterations, n=%d)" % n)
print("=" * 100)
rows = []
for col, lab in {**METRICS, **SUPPORT}.items():
    for tag, d in [('A1', d1), ('A2', d2)]:
        x = d[col].values
        rows.append(dict(Metric=lab, Approach=tag, n=len(x), Mean=x.mean(), SD=x.std(ddof=1),
                         Min=x.min(), Q1=np.percentile(x, 25), Median=np.median(x),
                         Q3=np.percentile(x, 75), Max=x.max(),
                         First=x[0], Final=x[-1],
                         CV_pct=100 * x.std(ddof=1) / abs(x.mean())))
desc = pd.DataFrame(rows)
print(desc.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
desc.to_csv(os.path.join(OUT, "table_descriptive_statistics.csv"), index=False)

print("\n" + "=" * 100)
print("3. PHASE-WISE MEANS (+/- SD)  and  REDUCTION FROM INITIAL")
print("=" * 100)
prows = []
for col, lab in METRICS.items():
    for tag, d in [('A1', d1), ('A2', d2)]:
        x = d[col].values
        r = dict(Metric=lab, Approach=tag, Initial=x[0])
        for pn, pi in ph.items():
            r[pn + ' mean'] = x[pi].mean()
            r[pn + ' sd'] = x[pi].std(ddof=1)
        r['Reduction_initial_to_late_%'] = 100 * (x[0] - x[ph['late (it 34-49)']].mean()) / abs(x[0])
        r['Plateau_CV_%'] = 100 * x[ph['late (it 34-49)']].std(ddof=1) / abs(x[ph['late (it 34-49)']].mean())
        prows.append(r)
phase = pd.DataFrame(prows)
print(phase.to_string(index=False, float_format=lambda v: f"{v:10.3f}"))
phase.to_csv(os.path.join(OUT, "table_phase_statistics.csv"), index=False)

print("\n" + "=" * 100)
print("4. CONVERGENCE DIAGNOSTICS")
print("=" * 100)


def exp_decay(t, a, k, c):
    return a * np.exp(-k * t) + c


conv = []
ts = d1['time/total_timesteps'].values.astype(float)
for col, lab in METRICS.items():
    for tag, d in [('A1', d1), ('A2', d2)]:
        x = d[col].values.astype(float)
        plateau = x[ph['late (it 34-49)']].mean()
        band = 0.10 * abs(plateau)
        inside = np.abs(x - plateau) <= band
        # first iteration after which the metric stays inside the +/-10% band
        k_conv = np.nan
        for i in range(len(x)):
            if inside[i:].all():
                k_conv = i
                break
        try:
            p0 = [x[0] - plateau, 5e-5, plateau]
            popt, _ = curve_fit(exp_decay, ts - ts[0], x, p0=p0, maxfev=60000)
            a_, k_, c_ = popt
            yhat = exp_decay(ts - ts[0], *popt)
            r2 = 1 - np.sum((x - yhat) ** 2) / np.sum((x - x.mean()) ** 2)
            half = np.log(2) / k_ if k_ > 0 else np.nan
        except Exception as e:
            a_ = k_ = c_ = r2 = half = np.nan
        rho, pv = stats.spearmanr(ts, x)
        conv.append(dict(Metric=lab, Approach=tag, Plateau_mean=plateau,
                         Conv_iter=(k_conv + 2 if not np.isnan(k_conv) else np.nan),
                         Conv_timestep=(ts[int(k_conv)] if not np.isnan(k_conv) else np.nan),
                         Exp_asymptote=c_, Exp_halflife_steps=half, Exp_R2=r2,
                         Spearman_rho=rho, Spearman_p=pv))
convdf = pd.DataFrame(conv)
print(convdf.to_string(index=False, float_format=lambda v: f"{v:12.4f}"))
convdf.to_csv(os.path.join(OUT, "table_convergence_diagnostics.csv"), index=False)

print("\n" + "=" * 100)
print("5. A1 vs A2 COMPARISON  (plateau = last third, iterations 34-49, n=16)")
print("=" * 100)


def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (nx * ny)


pi = ph['late (it 34-49)']
comp = []
for col, lab in {**METRICS, **SUPPORT}.items():
    x = d1[col].values[pi]
    y = d2[col].values[pi]
    w_s, w_p = stats.wilcoxon(x, y)          # paired by training checkpoint
    u_s, u_p = stats.mannwhitneyu(x, y, alternative='two-sided')
    t_s, t_p = stats.ttest_rel(x, y)
    delta = cliffs_delta(x, y)
    # bootstrap CI of the median difference (paired)
    rng = np.random.default_rng(42)
    diffs = x - y
    bs = [np.median(rng.choice(diffs, len(diffs), replace=True)) for _ in range(20000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    comp.append(dict(Metric=lab, A1_mean=x.mean(), A1_sd=x.std(ddof=1),
                     A2_mean=y.mean(), A2_sd=y.std(ddof=1),
                     Diff_A2_minus_A1=y.mean() - x.mean(),
                     Rel_change_pct=100 * (y.mean() - x.mean()) / abs(x.mean()),
                     Median_diff_A1_minus_A2=np.median(diffs), CI95_lo=lo, CI95_hi=hi,
                     Wilcoxon_W=w_s, Wilcoxon_p=w_p, MWU_p=u_p, ttest_p=t_p,
                     Cliffs_delta=delta))
cmp = pd.DataFrame(comp)
print(cmp.to_string(index=False, float_format=lambda v: f"{v:12.4f}"))
cmp.to_csv(os.path.join(OUT, "table_approach_comparison.csv"), index=False)

print("\n" + "=" * 100)
print("6. STRUCTURE OF THE TOTAL LOSS  (is it an independent signal?)")
print("=" * 100)
for tag, d in [('A1', d1), ('A2', d2)]:
    tot = d['train/loss'].values
    vl = d['train/value_loss'].values
    pg = d['train/policy_gradient_loss'].values
    ent = d['train/entropy_loss'].values
    r_v = stats.pearsonr(tot, vl)
    r_e = stats.pearsonr(tot, ent)
    ratio = tot / vl
    print(f"\n{tag}: corr(total, value_loss) r={r_v[0]:.4f} (p={r_v[1]:.2e}) | "
          f"corr(total, entropy_loss) r={r_e[0]:.4f} (p={r_e[1]:.2e})")
    print(f"    total/value_loss ratio: mean={ratio.mean():.4f} sd={ratio.std(ddof=1):.4f} "
          f"[theoretical vf_coef = 0.5]")
    print(f"    |policy_gradient_loss| mean = {np.abs(pg).mean():.5f}  "
          f"({100*np.abs(pg).mean()/np.abs(tot).mean():.4f}% of |total loss|)")
    print(f"    ent_coef contribution: 0 (entropy term not in logged loss; ent_coef=0 default)")

print("\n" + "=" * 100)
print("7. POLICY ENTROPY IN INTERPRETABLE UNITS (5 discrete actions, H_max = ln5 = %.4f nats)" % np.log(5))
print("=" * 100)
Hmax = np.log(5)
for tag, d in [('A1', d1), ('A2', d2)]:
    H = -d['train/entropy_loss'].values
    print(f"{tag}: H_initial={H[0]:.4f} nats ({100*H[0]/Hmax:5.1f}% of max) -> "
          f"H_final={H[-1]:.4f} nats ({100*H[-1]/Hmax:5.1f}% of max); "
          f"plateau mean={H[pi].mean():.4f} ({100*H[pi].mean()/Hmax:.1f}%); "
          f"total decay={100*(H[0]-H[-1])/H[0]:.1f}%")
    # effective number of actions
    print(f"     perplexity exp(H): {np.exp(H[0]):.3f} -> {np.exp(H[-1]):.3f} effective actions")
    sl = stats.linregress(ts, H)
    print(f"     linear entropy decay rate: {sl.slope*1e4:.4f} nats / 10k steps (R2={sl.rvalue**2:.3f})")

print("\n" + "=" * 100)
print("8. CRITIC QUALITY: implied return variance and residual decomposition")
print("=" * 100)
for tag, d in [('A1', d1), ('A2', d2)]:
    vl = d['train/value_loss'].values
    ev = d['train/explained_variance'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        implied_var = vl / (1 - ev)
    print(f"{tag}: plateau value_loss={vl[pi].mean():8.2f} | plateau EV={ev[pi].mean():.4f} | "
          f"implied Var(returns)={np.nanmean(implied_var[pi]):8.2f} "
          f"(median {np.nanmedian(implied_var[pi]):8.2f})")

print("\n" + "=" * 100)
print("9. TRAINING STABILITY / TRUST-REGION HEALTH")
print("=" * 100)
for tag, d in [('A1', d1), ('A2', d2)]:
    kl = d['train/approx_kl'].values
    cf = d['train/clip_fraction'].values
    print(f"{tag}: approx_KL  max={kl.max():.5f} mean={kl.mean():.5f} plateau={kl[pi].mean():.5f} "
          f"| clip_fraction max={cf.max():.4f} plateau={cf[pi].mean():.4f} "
          f"| monotonic reward improvement (Spearman rho vs steps) = "
          f"{stats.spearmanr(ts, d['rollout/ep_rew_mean'].values)[0]:.3f}")

print("\n" + "=" * 100)
print("10. LEARNING-CURVE (REWARD) SUMMARY")
print("=" * 100)
for tag, d, raw in [('A1', d1, a1), ('A2', d2, a2)]:
    r = raw['rollout/ep_rew_mean'].values
    print(f"{tag}: R_initial={r[0]:8.2f} R_final={r[-1]:7.2f} R_max={r.max():7.2f} "
          f"plateau(last16)={r[-16:].mean():7.2f} +/- {r[-16:].std(ddof=1):.2f} | "
          f"crossing R=0 at ~{int(raw.loc[(r>0).argmax(),'time/total_timesteps'])} steps")

summary = dict(
    n_iterations=int(n), timesteps=int(ts[-1]),
    a1_plateau={k: float(d1[k].values[pi].mean()) for k in METRICS},
    a2_plateau={k: float(d2[k].values[pi].mean()) for k in METRICS},
)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved tables to", OUT)

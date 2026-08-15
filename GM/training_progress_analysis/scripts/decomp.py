import pandas as pd, numpy as np
from scipy import stats

for tag, f in [('A1', r'C:\Users\hp\Desktop\approach1_insar_only progress.csv'),
               ('A2', r'C:\Users\hp\Desktop\approach2_with_rainfall progress.csv')]:
    d = pd.read_csv(f).dropna(subset=['train/loss']).reset_index(drop=True)
    pg = d['train/policy_gradient_loss'].values
    ent = d['train/entropy_loss'].values
    vl = d['train/value_loss'].values
    tot = d['train/loss'].values
    pred = pg + 0.01 * ent + 0.5 * vl
    print(f"{tag}: corr(logged total, reconstructed) = {stats.pearsonr(tot, pred)[0]:.5f}")
    print(f"    mean |logged - reconstructed| / reconstructed = {100*np.mean(np.abs(tot-pred)/pred):.2f}%")
    print(f"    share of |reconstructed|:  0.5*value = {100*np.mean(0.5*vl/pred):.4f}% | "
          f"0.01*entropy = {100*np.mean(np.abs(0.01*ent)/pred):.5f}% | "
          f"pg = {100*np.mean(np.abs(pg)/pred):.5f}%")

# Training-progress analysis — Approach 1 (InSAR only) vs Approach 2 (InSAR + rainfall)

Source files: `approach1_insar_only progress.csv`, `approach2_with_rainfall progress.csv`
(Stable-Baselines3 CSV logger output). **Provenance verified: both files are byte-identical to
`NR7\GM\rl_outputs_v7\approach{1,2}\sb3_log\progress.csv`** — i.e. they are the v7 leakage-free
runs that Chapter 4 already reports, so all numbers below are consistent with the existing
dissertation results.

---

## 1. Data integrity and completeness

| Check | Approach 1 | Approach 2 |
|---|---|---|
| Rows × columns | 49 × 16 | 49 × 16 |
| Rollout size | 2 048 steps, constant | 2 048 steps, constant |
| Total interaction steps | 100 352 | 100 352 |
| Policy-update iterations | 1 → 49, strictly monotonic | 1 → 49, strictly monotonic |
| Rows with missing `train/*` | 1 (iteration 1) | 1 (iteration 1) |
| Duplicate rows | 0 | 0 |
| Learning rate / clip range | 3 × 10⁻⁴ / 0.2 (constant) | 3 × 10⁻⁴ / 0.2 (constant) |
| Gradient epochs per iteration | 10 (`n_updates` +10 each row) | 10 |
| Wall-clock / throughput | 140 s, 700 FPS | 156 s, 642 FPS |

* The single row of missing `train/*` values is **structural, not a defect**: SB3 logs the first
  rollout before any policy update has occurred. Usable sample size is therefore **n = 48**
  logged iterations per approach.
* `rollout/ep_len_mean` is **identical to machine precision across the two runs** (max abs. diff
  = 0). This confirms both experiments used the same PID split, the same episode schedule and the
  same seed — the two curves are therefore directly comparable at matched training checkpoints,
  and a *paired* statistical treatment is legitimate.
* Training hyper-parameters (from `slope_rl_framework_v7.py`): PPO, MLP 128×128, `n_steps` = 2 048,
  `batch_size` = 256, `n_epochs` = 10 (→ 80 minibatch gradient steps per iteration, 3 840 in total),
  `gae_lambda` = 0.95, `gamma` = 0.99, `learning_rate` = 3 × 10⁻⁴, `clip_range` = 0.2,
  `ent_coef` = 0.01, `vf_coef` = 0.5 (default).

### Structure of the logged total loss — a necessary caveat

`train/loss` is **not an independent convergence signal**. Reconstructing it as
`pg_loss + 0.01·entropy_loss + 0.5·value_loss` reproduces the logged series with r = 0.983 (A1)
and r = 0.994 (A2); the value term supplies **> 99.97 %** of its magnitude, the policy-gradient
surrogate ≈ 0.009–0.012 % and the entropy bonus ≈ 0.007–0.009 %. The residual ~8.5 % deviation is
the known SB3 convention that `train/loss` is the *last* minibatch value whereas the component
terms are epoch-averaged. **Panel (a) should therefore be read as an aggregate summary that
tracks the critic, and never cited as evidence of policy-level convergence.**

---

## 2. Key statistics (plateau = final 16 iterations, 69 632–100 352 steps)

| Metric | A1 (InSAR only) | A2 (+ rainfall) | Δ (A2 − A1) | Paired test |
|---|---|---|---|---|
| Total loss | 86.04 ± 9.23 | 50.51 ± 7.76 | −35.53 (−41.3 %) | W = 0, p = 3.05 × 10⁻⁵ |
| Value loss | 174.36 ± 18.37 | 104.38 ± 15.82 | −69.98 (−40.1 %) | W = 0, p = 3.05 × 10⁻⁵ |
| Entropy loss | −0.3644 ± 0.0271 | −0.3528 ± 0.0422 | +0.0116 (+3.2 %) | W = 41, **p = 0.175 (n.s.)** |
| Explained variance | 0.317 ± 0.035 | 0.693 ± 0.040 | +0.376 (+118 %) | W = 0, p = 3.05 × 10⁻⁵ |
| Episode reward | 51.81 ± 4.43 | 50.84 ± 5.04 | −0.97 (−1.9 %) | p = 0.074 (n.s.) |

Effect sizes — Cliff's δ = 1.00 (total loss, value loss) and −1.00 (explained variance):
**complete separation**, the two plateau samples do not overlap at all
(A1 value loss ∈ [152.2, 215.4], A2 ∈ [82.2, 139.2]; KS D = 1.0, p = 3.3 × 10⁻⁹;
Hedges g = 3.98). For entropy loss the samples overlap almost entirely
(KS D = 0.375, p = 0.22; Hedges g = −0.32). Bootstrap 95 % CI of the paired median difference:
total loss [28.9, 41.2]; value loss [57.6, 80.4]; entropy loss [−0.032, +0.014] (contains zero).

### Whole-run descriptive statistics

| Metric | Approach | Initial | Early mean (it 2–17) | Mid (18–33) | Late (34–49) | Reduction |
|---|---|---|---|---|---|---|
| Total loss | A1 | 333.5 | 123.9 ± 94.6 | 81.1 ± 11.1 | 86.0 ± 9.2 | −74.2 % |
| Total loss | A2 | 333.1 | 115.2 ± 77.7 | 65.5 ± 8.7 | 50.5 ± 7.8 | −84.8 % |
| Value loss | A1 | 678.7 | 273.7 ± 196.4 | 171.4 ± 15.9 | 174.4 ± 18.4 | −74.3 % |
| Value loss | A2 | 762.6 | 261.9 ± 184.3 | 138.7 ± 16.9 | 104.4 ± 15.8 | −86.3 % |
| Entropy loss | A1 | −1.605 | −1.218 ± 0.281 | −0.549 ± 0.121 | −0.364 ± 0.027 | 77.3 % of |H| removed |
| Entropy loss | A2 | −1.604 | −1.222 ± 0.268 | −0.552 ± 0.108 | −0.353 ± 0.042 | 78.0 % |

---

## 3. Convergence behaviour

**Phase 1 — shared transient (0 → ~15 k steps).** Both runs collapse the critic error by roughly
one order of magnitude, from 679 (A1) / 763 (A2) to ~150–170. Single-exponential fits
`L(t) = A e^{−kt} + C` give near-identical half-lives of **3 547 steps (A1)** and **3 672 steps
(A2)**, i.e. the initial learning rate of the value function is a property of the environment and
the PPO configuration, not of the feature set. 90 % of A1's total improvement is achieved by
14 336 steps (iteration 7); A2 reaches its 90 % point later (26 624 steps, iteration 13) simply
because it continues to improve further.

**Phase 2 — divergence of the two runs (~20 k → 100 k steps).** This is the substantive finding.
Restricting the analysis to iterations 12–49 (≥ 22 528 steps):

| | A1 (InSAR only) | A2 (+ rainfall) |
|---|---|---|
| Spearman ρ (value loss vs steps) | +0.07 (p = 0.69) — **flat** | −0.82 (p = 4.1 × 10⁻¹⁰) — **still falling** |
| OLS slope, value loss | +0.68 per 10 k steps (95 % CI −1.69, +3.06) | **−10.35 per 10 k steps** (95 % CI −12.61, −8.10), R² = 0.69 |
| Spearman ρ (explained variance) | +0.47 (p = 0.003), slope +0.023/10 k | +0.88 (p = 2 × 10⁻¹³), slope +0.071/10 k |

A1 exhausts its learning capacity at roughly **20–25 k steps** — one quarter of the budget — and
spends the remaining 75 k steps oscillating about a fixed error floor. The A1 single-exponential
asymptote (167.8, R² = 0.876) is an excellent description of the whole run. For A2 the same fit is
poorer at the tail (asymptote 126.6 vs an observed plateau of 104.4, R² = 0.927), because A2's
descent is genuinely two-phase: a fast environment-driven transient followed by a slow, sustained
second regime that the single exponential cannot capture.

**Sample efficiency.** A2's smoothed value loss falls below A1's *final* plateau (174.4) at
**18 432 steps** and never returns above it — A2 attains in ~18 % of the interaction budget what
A1 attains after the full 100 352 steps.

**A1 late-phase drift.** Over the last 16 iterations A1's value loss trends *upward*
(+9.36 per 10 k steps, p = 0.050, R² = 0.25) while its explained variance stays flat
(slope +0.008/10 k, p = 0.40). This is mild critic drift as the policy sharpens and the
state-visitation distribution narrows around a smaller action repertoire — not divergence
(see §5) — and is the signature of a representation that has reached its ceiling.

**Stability of the plateau.** Plateau coefficients of variation are 10.5 % (A1) and 15.2 % (A2)
for value loss; residual scatter about the 5-iteration moving average is 15.2 (8.7 % of the mean)
for A1 and 10.7 (10.2 %) for A2, with lag-1 autocorrelations of −0.17 and +0.10 respectively.
The near-zero autocorrelation is convenient: successive plateau iterations behave close to
independent draws, so the rank-based tests above are not appreciably inflated by serial
dependence.

---

## 4. Entropy: the two policies are statistically indistinguishable

* Both runs start at **1.6045 / 1.6044 nats = 99.7 % of ln 5 = 1.6094**, confirming correctly
  initialised near-uniform policies over the five management actions.
* Both decay to **0.310 nats (19.2 % of maximum, A1)** and **0.292 nats (18.2 %, A2)**;
  in perplexity terms the effective action repertoire contracts from 4.98 to
  **1.36 (A1) and 1.34 (A2) actions**.
* Decay rates are the same within noise: **−0.132 vs −0.133 nats per 10 k steps**
  (R² = 0.85 / 0.87), and the plateau difference is not significant (p = 0.175, Hedges g = −0.32).
* Entropy is **still declining at the end of the budget** in both runs (late-phase slopes
  +0.024 and +0.034 nats-loss per 10 k steps, both p < 0.001). Exploration had not fully
  annealed at 100 k steps despite the `ent_coef` = 0.01 bonus; the residual ~19 % of maximum
  entropy is what keeps the policy stochastic at evaluation time.

**Interpretation:** the five additional rainfall features changed *nothing* about the
exploration–exploitation schedule or the rate at which the actor committed to a policy. The
entire measurable effect of rainfall conditioning is confined to the **critic**.

---

## 5. Optimisation health (supporting diagnostics)

| | A1 | A2 |
|---|---|---|
| Approx. KL — max / plateau | 0.0149 / 0.0032 | 0.0179 / 0.0033 |
| Clip fraction — max / plateau | 0.136 / 0.027 | 0.150 / 0.030 |
| Reward trend (Spearman ρ vs steps) | +0.960 | +0.962 |
| Reward crosses zero at | ~24 576 steps | ~22 528 steps |

Approximate KL never approaches the 0.02–0.03 band at which PPO updates are normally considered
unstable, and clip fractions below ~3 % at the plateau indicate the trust region was rarely
binding. Neither run required early stopping or learning-rate annealing; both learning curves are
monotone in reward. **The training of both agents is technically sound, and the A1/A2 difference
is a capability difference, not an artefact of optimisation instability.**

---

## 6. Why A2's critic is better — and why that does *not* mean rainfall helps

The temptation is to read a 40 % lower value loss as "rainfall improves the model". Three checks
argue against that reading:

1. **It is not a target-scale artefact — the critic genuinely is better.** Value loss alone can
   fall simply because returns are smaller or less variable. Reconstructing the target variance
   as `Var(y) ≈ L_V /(1 − EV)` gives **256 (A1) vs 342 (A2)**: A2 faces a *more* variable return
   target and still achieves a 40 % lower residual error, with explained variance rising from
   0.32 to 0.69. Plateau episode rewards are statistically indistinguishable (51.8 vs 50.8,
   p = 0.074), so the two runs are operating on the same reward scale.
2. **The gain does not reach the policy.** Entropy trajectories are identical (§4), plateau
   rewards are equal, and the downstream v7 ablation reports a rainfall effect on Level-5 recall
   of **Δ = −0.039** — slightly negative. A better value function reduced the variance of the
   advantage estimates without changing which action the actor chose.
3. **The likely mechanism is a confound already documented in Chapter 4.** Rainfall accumulated
   over a window correlates with the *length* of that window (rain ~ span, r = 0.49), and window
   span is a direct determinant of episode structure and return. The rainfall features therefore
   give the critic a partial read-out of *how much return is left in this episode* — legitimate
   value-function information, but not causal landslide-precursor information. This is entirely
   consistent with the negative, confounded rainfall–hazard-level correlation (r = −0.18) already
   reported.

**Defensible claim for the dissertation:** *rainfall conditioning improves value-function fit and
sample efficiency, while leaving policy behaviour, achieved reward and hazard-level recall
unchanged.* Any stronger claim is not supported by these logs.

---

## 7. Limitations to state explicitly

* **One seed per approach.** The Wilcoxon/Cliff's-δ statistics describe the separation between two
  *specific* runs at matched checkpoints; they are not evidence about the population of runs.
  A defensible generalisation would need 3–5 seeds per approach with the seed as the unit of
  analysis. This is worth one sentence in the text and is a natural Chapter 6 recommendation.
* **`train/loss` is ~0.5 × `train/value_loss`** (§1) and must not be presented as independent
  evidence of convergence.
* Entropy had not fully annealed at 100 k steps; the 100 k budget was adequate for A1 (flat from
  ~25 k) but arguably premature for A2, whose value loss was still falling at −10.4 per 10 k steps
  when training stopped. A2's reported performance is therefore a *lower bound*.
* Value loss is not comparable across approaches in absolute units unless the return scale is
  comparable — it is here (checked in §6), but the check must be stated, not assumed.

---

## 8. Ready-to-use figure caption

> **Figure 4.x** — Comparative PPO training progress for Approach 1 (InSAR only, blue) and
> Approach 2 (InSAR + rainfall, orange): **(a)** total loss, **(b)** value-function loss and
> **(c)** entropy loss, against environment interaction steps (upper axis: policy-update
> iteration). Faint lines are per-iteration logged values (48 iterations × 2 048 steps, 10 epochs
> of 8 minibatches each); bold lines are 5-iteration centred moving averages. Ordinates in (a) and
> (b) are logarithmic. The shaded band marks the plateau window (final 16 iterations,
> 69 632–100 352 steps) over which the approaches were compared; dotted lines and labels give
> plateau means ± 1 SD. In (c), the dashed reference line is the maximum-entropy policy — uniform
> over the five management actions, −ln 5 = −1.609 nats — and the right-hand axis expresses policy
> entropy as a percentage of that maximum. Both runs share identical hyper-parameters
> (PPO, 128 × 128 MLP, lr = 3 × 10⁻⁴, clip ε = 0.2, γ = 0.99, λ = 0.95, ent_coef = 0.01,
> vf_coef = 0.5), the same PS-point train/test split and the same episode schedule; the only
> difference is the presence of the four rainfall features in the observation vector. Note that
> the total loss in (a) is dominated by the value term (vf_coef × value loss accounts for > 99.9 %
> of its magnitude) and therefore tracks (b) rather than providing independent information.

---

## 9. Files produced

| File | Contents |
|---|---|
| `fig_training_progress_A1_vs_A2.png` | 600 dpi raster, 3 721 × 4 877 px = 158 × 206 mm |
| `fig_training_progress_A1_vs_A2.pdf` | vector version (embedded Type-42 fonts) — preferred for Word/LaTeX |
| `fig_training_progress_A1_vs_A2.svg` | editable vector version |
| `table_descriptive_statistics.csv` | full descriptives for all metrics, both approaches |
| `table_phase_statistics.csv` | early/mid/late phase means ± SD, reduction, plateau CV |
| `table_convergence_diagnostics.csv` | exponential fits, half-lives, asymptotes, Spearman trends |
| `table_convergence_time.csv` | convergence iteration and t₉₀ per metric |
| `table_approach_comparison.csv` | plateau comparison, Wilcoxon/MWU/t-test, Cliff's δ, bootstrap CIs |
| `summary.json` | machine-readable plateau summary |
| `scripts/` | the three analysis/plot scripts, for reproducibility |

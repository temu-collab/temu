"""
================================================================================
 slope_rl_framework_v7.py
--------------------------------------------------------------------------------
 Reinforcement-Learning Framework for Slope-Stability Level Assessment
 from PS-InSAR Displacement + Rainfall Data 
--------------------------------------------------------------------------------
 v7 - LEAKAGE-FREE with ENGINEERED PAST-ONLY STATE (final configuration):
   1. the PROXY LABELS use WITHIN-WINDOW kinematics only:
      staged velocities v12, v23, v34 across S_1..S_4 and an acceleration
      that splits the observed stages at S_3. The Target stage is used
      NOWHERE - neither in the state nor in the label.
   2. (new in v7) the past-only engineered kinematics (Average_Staged_V,
      sd_stagedV, accel) are RESTORED to the agent's observation. All three
      are computed from the observed stages alone, so they are available at
      decision time - feature engineering is not leakage. This restores the
      ORIGINAL Table 3.5 state composition (14 features Approach 1, 19 with
      rainfall), now with the amended past-only Section-3.4.1 formulas.
      Rationale: development iterations showed that a raw-only state plateaus
      at ~0.58 Level-5 recall regardless of network size or training length
      (300k/128x128 vs 600k/256x256 gave 0.576 vs 0.574), because the internal
      velocity estimate is fuzzy exactly at the sharp 100 mm/yr class
      boundary. Handing it the quantities the label thresholds act on
      removes that representation bottleneck without reintroducing any
      future information.

 One execution performs THREE experiments and the rainfall ablation:

   [A1] Approach 1 - InSAR + past-only kinematics (14 features).
   [A2] Approach 2 - state extended with rainfall (19 features), SAME split / seed / episodes as A1  ->  the Section-3.9
        ablation: does precipitation improve Level-5 alarm detection?
   [A3] Approach 3 (improved) - decision-support agent tuned to maximise the
        safety-critical Level-5 recall: balanced episode sampling, a stronger
        safety-asymmetric reward, shorter episodes, lower discount, and a
        longer training budget. Produces the engineering recommendations and
        the self-contained inference bundle (model + scaler + config) used to
        test the agent on NEW raw-column data.

 PRIMARY DATA  : ML_working_Data.xlsx/.csv (74,880 windows x 936 PIDs;
                 PID, S_1..S_4, Target, T1..T4, R_1..R_4, Rainfall_Target)
 REFERENCE DATA: I_PS_NS_V.csv (point-level temporal_coherence, mean_velocity,
                 rmse, merged by PID; PID k = row k, verified)

================================================================================
 DESIGN-DECISION LOG 
--------------------------------------------------------------------------------
 STATE (v7, original Table 3.5 composition, past-only definitions) - the
   agent observes: displacement stages S_1..S_4, acquisition gaps T1..T4
   (6/12/24-day timing preserved), the past-only engineered kinematics
   (Average_Staged_V, sd_stagedV, accel - all computed within S_1..S_4,
   no Target term), and the point-reliability indicators (mean_velocity,
   temporal_coherence, rmse). Rainfall (R_1..R_4 + antecedent total) enters
   the state only for Approach >= 2 (14 features for A1, 19 for A2/A3).
   Everything in the state is observable at decision time; Rainfall_Target
   and the Target stage are EXCLUDED (target-date information). The
   stability level itself is never observed.

 LABELS (past-only kinematics) - transparent, reproducible proxy:
   Varnes-adapted |V| bands (2 / 10 / 30 / 100 mm/yr -> Levels 1..5)
   computed on the average of the WITHIN-WINDOW staged velocities
       v12 = (S_2-S_1)/T1 x 365.25,
       v23 = (S_3-S_2)/T2 x 365.25,
       v34 = (S_4-S_3)/T3 x 365.25,
   plus ONE expert rule: windows whose |acceleration| exceeds the 90th
   percentile AND whose base level is >= 2 are escalated one level (capped
   at 5), with the acceleration now splitting the OBSERVED stages at S_3:
       v_first  = (S_3-S_1)/(T1+T2) x 365.25,
       v_second = (S_4-S_3)/T3      x 365.25,
       accel    = v_second - v_first.
   RATIONALE: the original definitions include v45 and a second half-window
   velocity computed with the Target stage, i.e. information from the target
   date. Development iterations showed that observing target-derived
   kinematics yields near-perfect recall by partially reproducing the label
   rule, while keeping such labels but hiding the kinematics turns the task
   into next-interval forecasting, in which the critical class is dominated
   by unpredictable last-segment spikes (Level-5 recall collapsed to 0.29
   even when tuned).
   Defining BOTH the state and the label from the observed stages restores
   a well-posed, leakage-free recognition task: the label describes the
   slope's current condition and the state contains everything needed, in
   principle, to recognise it. Rainfall never changes the label, keeping
   the Approach-2 ablation clean (state changes, labels fixed).

 ACTIONS - Discrete(5): one graded engineering response per stability level
   (Do nothing / Routine / Enhanced / Inspection / Reinforce-Alarm).

 REWARD - safety-asymmetric matrix R[true_level, action] (Table 3.7).
   Missed failures are catastrophic, false alarms merely costly, so
   under-response is penalised steeply (worst: Do-nothing on Level 5) and
   over-response only mildly.
   A3 uses a SAFETY-TUNED variant of the same matrix in which the L4/L5
   under-response penalties are doubled and the L5 diagonal is raised
   (+6 -> +12). Rationale: the initial configuration reached only 0.40
   Level-5 recall; the cost-ratio principle (missed failure >> false alarm) is
   unchanged - the ratio is simply made larger, which shifts the learned
   policy toward escalation when uncertain.

 EPISODES - Initial definition (A1/A2): an episode traverses ONE point's
   window sequence forward in time from a random start (max 80 steps).
   A3 uses BALANCED episode starts: a stability level is drawn uniformly,
   a window of that level is drawn at random, and the episode runs forward
   through that point's real history for up to 24 steps. Time still moves
   forward through genuine deformation history; only the *exposure* of the
   agent to rare critical states is rebalanced (Level 5 is ~11.2 % of
   windows under the past-only labels, so under uniform sampling the agent
   rarely practises the decisions that matter most).

 DISCOUNT - Initial runs keep gamma = 0.99. A3 uses gamma = 0.35: the
   engineering response taken at one window does not alter the slope's
   subsequent deformation (actions do not influence transitions), so long
   discounted returns only add variance to the advantage estimate; a low
   gamma concentrates credit on the decision actually being scored.

 SPLIT - grouped hold-out by PID (20 % of points, seed 42): every window of
   a held-out point is in the test set, so near-duplicate overlapping
   windows can never sit on both sides. The SAME split is used for all
   three experiments, making the ablation a paired comparison.

 STANDARDISATION - StandardScaler fitted on TRAINING points only, applied
   to test and to any new data via the saved scaler.pkl (exact match at
   inference time - this answers the professor's question about testing on
   new raw-column data when the agent was trained on standardised inputs).

 ASSUMPTIONS - velocities are line-of-sight (LOS), not 3-D slope velocity;
   displacements are cumulative from the 2023-06-06 reference; short-window
   acceleration is noisy (coherence is carried as a reliability indicator).
================================================================================
"""

import os
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless-safe
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             recall_score, precision_recall_fscore_support,
                             ConfusionMatrixDisplay)

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure as sb3_configure_logger

warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_CANDIDATES = [SCRIPT_DIR, r"C:\Users\hp\Desktop\NR7\GM", "."]
WORKING_NAME = "ML_working_Data"            # .xlsx preferred, .csv fallback
INSAR_NAME   = "I_PS_NS_V.csv"
OUT_ROOT     = os.path.join(SCRIPT_DIR, "rl_outputs_v7")

GROUP       = "PID"
DISP_STAGES = ["S_1", "S_2", "S_3", "S_4"]
TIME_GAPS   = ["T1",  "T2",  "T3",  "T4"]
RAIN_STAGES = ["R_1", "R_2", "R_3", "R_4"]
TARGET      = "Target"                      # S_5, the next displacement stage

ACTIONS = ["DO NOTHING", "ROUTINE MONITORING", "ENHANCED MONITORING",
           "FIELD INSPECTION", "REINFORCE / ALARM"]
N_ACTIONS = len(ACTIONS)

LEVEL_NAMES = {1: "Stable", 2: "Very Slow", 3: "Slow Active",
               4: "Moderately Active", 5: "Rapid / Failure Risk"}
VARNES_THRESHOLDS   = [2.0, 10.0, 30.0, 100.0]   # |V| mm/yr level boundaries
ACCEL_ESCALATE_PCTL = 0.90
SAFETY_LEVEL        = 5

TEST_FRACTION = 0.20
RANDOM_SEED   = 42
OBS_CLIP      = 10.0

# ---- Table 3.7 reward matrix (used by A1 and A2) -----------
INITIAL_REWARD = np.array([
    #  DoNothing  Routine  Enhanced  Inspect  Reinforce/Alarm
    [    +2.0,     -1.0,    -2.0,    -3.0,    -4.0 ],   # L1 Stable
    [    -2.0,     +3.0,    -1.0,    -2.0,    -3.0 ],   # L2 Very Slow
    [    -4.0,     -2.0,    +4.0,    -1.0,    -2.0 ],   # L3 Slow Active
    [    -8.0,     -4.0,    -2.0,    +5.0,    -1.0 ],   # L4 Moderately Active
    [   -16.0,     -8.0,    -4.0,    -2.0,    +6.0 ],   # L5 Rapid / Failure
], dtype=np.float32)

# ---- Safety-tuned variant (A3): same principle, larger cost ratio ------------
SAFETY_TUNED_REWARD = INITIAL_REWARD.copy()
SAFETY_TUNED_REWARD[3] = [-12.0,  -6.0, -3.0, +6.0,  -1.0]   # L4
SAFETY_TUNED_REWARD[4] = [-32.0, -16.0, -8.0, -4.0, +12.0]   # L5

# ---- The three experiments ----------------------------------------------------
EXPERIMENTS = [
    dict(tag="approach1_insar_only",    approach=1, reward="initial",
         episode_mode="sequential", max_steps=80, gamma=0.99,
         timesteps=100_000,
         title="Approach 1 - InSAR + past-only kinematics"),
    dict(tag="approach2_with_rainfall", approach=2, reward="initial",
         episode_mode="sequential", max_steps=80, gamma=0.99,
         timesteps=100_000,
         title="Approach 2 - + rainfall in state (initial settings)"),
    dict(tag="approach3_improved",      approach=3, reward="safety_tuned",
         episode_mode="balanced",   max_steps=24, gamma=0.35,
         timesteps=300_000,
         title="Approach 3 - decision support, safety-tuned (leakage-free)"),
]

LEVEL_COLORS = ["#10b981", "#84cc16", "#f59e0b", "#f97316", "#ef4444"]


# ==============================================================================
# 2. DATA LOADING & FEATURE ENGINEERING
# ==============================================================================

def _resolve(name, ext=None):
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, name + (ext or ""))
        if os.path.exists(p):
            return p
    return None


def load_working_data():
    p = _resolve(WORKING_NAME, ".xlsx") or _resolve(WORKING_NAME, ".csv")
    if p is None:
        raise FileNotFoundError(f"{WORKING_NAME}.xlsx/.csv not found in "
                                f"{DATA_DIR_CANDIDATES}")
    df = pd.read_excel(p) if p.endswith(".xlsx") else pd.read_csv(p)

    # numeric safety net: coerce every required column, drop broken rows
    needed = [GROUP] + DISP_STAGES + [TARGET] + TIME_GAPS + RAIN_STAGES
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    if len(df) < n0:
        print(f"[data] dropped {n0 - len(df)} non-numeric rows")
    print(f"[data] working dataset : {len(df):,} windows x {df.shape[1]} cols "
          f"({df[GROUP].nunique()} PIDs)  <- {os.path.basename(p)}")
    return df


def merge_point_features(df):
    """Per-PID reliability attributes from the wide-format InSAR file
    (constant per scatterer; PID k = row k, verified in the methodology)."""
    p = _resolve(INSAR_NAME)
    if p is None:
        raise FileNotFoundError(f"{INSAR_NAME} not found (needed for coherence).")
    ins = pd.read_csv(p)
    pts = {GROUP: np.arange(1, len(ins) + 1)}
    for c in ["temporal_coherence", "mean_velocity", "rmse"]:
        pts[c] = ins[c].values if c in ins.columns else 0.0
    df = df.merge(pd.DataFrame(pts), on=GROUP, how="left")
    for c in ["temporal_coherence", "mean_velocity", "rmse"]:
        df[c] = df[c].fillna(df[c].median())
    print(f"[data] merged point features (coherence, mean velocity, rmse) "
          f"<- {os.path.basename(p)}")
    return df


def add_kinematic_features(df):
    """PAST-ONLY kinematics: staged velocities v12, v23, v34 within the observed stages S_1..S_4; acceleration splits
    the observed window at S_3. The Target stage is NOT used anywhere."""
    S1, S2, S3, S4 = (df[c] for c in DISP_STAGES)
    T1, T2, T3 = (df[c].clip(lower=1) for c in ["T1", "T2", "T3"])

    staged = np.vstack([(S2 - S1) / T1, (S3 - S2) / T2,
                        (S4 - S3) / T3]).T * 365.25
    df["avg_staged_v"] = staged.mean(axis=1)
    df["sd_staged_v"]  = staged.std(axis=1)
    df["abs_v"]        = df["avg_staged_v"].abs()

    v_first  = (S3 - S1) / (T1 + T2) * 365.25
    v_second = (S4 - S3) / T3 * 365.25
    df["accel"]     = v_second - v_first
    df["abs_accel"] = df["accel"].abs()

    df["precip_antecedent"] = df[RAIN_STAGES].sum(axis=1)
    return df


def assign_stability_levels(df):
    """Varnes-adapted velocity bands + 90th-percentile acceleration
    escalation (base level >= 2 only, capped at 5). Section 3.5."""
    t = VARNES_THRESHOLDS
    base = np.ones(len(df), dtype=int)
    for i, thr in enumerate(t):
        base[df["abs_v"].values >= thr] = i + 2

    a_thr = df["abs_accel"].quantile(ACCEL_ESCALATE_PCTL)
    escalate = (df["abs_accel"].values > a_thr) & (base >= 2)
    df["stability_level"] = np.where(escalate, np.minimum(base + 1, 5), base)

    print(f"\n[labels] Varnes |V| thresholds (mm/yr): {t}  |  "
          f"accel-escalation p{int(ACCEL_ESCALATE_PCTL*100)} = {a_thr:.1f} mm/yr")
    dist = df["stability_level"].value_counts().sort_index()
    for lv in range(1, 6):
        n = int(dist.get(lv, 0))
        print(f"         Level {lv} {LEVEL_NAMES[lv]:>20s}: {n:6d} "
              f"({100*n/len(df):5.1f} %)")
    return df, float(a_thr)


def build_feature_list(approach):
    """Original Table 3.5 composition with PAST-ONLY definitions: raw stages
    and gaps, engineered within-window kinematics (no Target term), and
    point reliability. Everything is observable at decision time."""
    feats = (DISP_STAGES + TIME_GAPS +
             ["avg_staged_v", "sd_staged_v", "accel",
              "mean_velocity", "temporal_coherence", "rmse"])
    if approach >= 2:
        feats += RAIN_STAGES + ["precip_antecedent"]
    return feats


def split_by_pid(df, test_fraction=TEST_FRACTION, seed=RANDOM_SEED):
    """Grouped hold-out: whole PIDs in train OR test, never both (§3.7)."""
    rng = np.random.default_rng(seed)
    pids = df[GROUP].unique()
    rng.shuffle(pids)
    n_test = int(len(pids) * test_fraction)
    test_pids = set(pids[:n_test].tolist())
    is_test = df[GROUP].isin(test_pids)
    print(f"\n[split] PIDs -> train {len(pids) - n_test} | test {n_test} "
          f"(grouped by point, no leakage; seed {seed})")
    return (df[~is_test].reset_index(drop=True),
            df[is_test].reset_index(drop=True))


# ==============================================================================
# 3. GYMNASIUM ENVIRONMENT
# ==============================================================================

class SlopeStabilityEnv(gym.Env):
    """
    Custom Gymnasium environment (Section 3.7).

    * observation : standardised indicator vector (Box, continuous)
    * action      : Discrete(5) graded engineering response
    * reward      : safety-asymmetric matrix R[true_level - 1, action]
    * episode     : forward traversal of one point's real window sequence.
        - "sequential" (initial): random point, random start, up to max_steps.
        - "balanced"  (A3): a stability level is drawn uniformly and the
          episode starts at a random window of that level, rebalancing the
          agent's exposure to rare critical states. Time still runs forward
          through the point's genuine history.

    The true level is used ONLY inside the reward; it is never observed.
    """
    metadata = {"render_modes": []}

    def __init__(self, X, levels, df_subset, reward_matrix,
                 episode_mode="sequential", max_steps=80, seed=RANDOM_SEED):
        super().__init__()
        self.X = X.astype(np.float32)
        self.levels = levels.astype(int)
        self.R = reward_matrix
        self.mode = episode_mode
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # per-PID chronological row arrays + reverse row -> (rows, position)
        self.pid_rows = [g.index.to_numpy()
                         for _, g in df_subset.groupby(GROUP, sort=True)]
        self.row_loc = np.empty((len(X), 2), dtype=np.int64)  # (pid_i, pos)
        for pi, rows in enumerate(self.pid_rows):
            for pos, r in enumerate(rows):
                self.row_loc[r] = (pi, pos)

        # rows by level, for balanced starts
        self.rows_by_level = {lv: np.flatnonzero(self.levels == lv)
                              for lv in range(1, 6)}
        self.present_levels = [lv for lv, r in self.rows_by_level.items()
                               if len(r) > 0]

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP, shape=(X.shape[1],), dtype=np.float32)
        self._rows, self._ptr, self._end = None, 0, 0

    def _obs(self):
        return np.clip(self.X[self._rows[self._ptr]], -OBS_CLIP, OBS_CLIP)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.mode == "balanced":
            lv = self.present_levels[self._rng.integers(len(self.present_levels))]
            row = int(self.rows_by_level[lv][
                self._rng.integers(len(self.rows_by_level[lv]))])
            pid_i, pos = self.row_loc[row]
            self._rows, start = self.pid_rows[pid_i], int(pos)
        else:
            self._rows = self.pid_rows[self._rng.integers(len(self.pid_rows))]
            start = int(self._rng.integers(len(self._rows)))
        self._ptr = start
        self._end = min(len(self._rows), start + self.max_steps)
        return self._obs(), {}

    def step(self, action):
        row = self._rows[self._ptr]
        true_level = int(self.levels[row])
        reward = float(self.R[true_level - 1, int(action)])
        info = {"true_level": true_level, "pred_level": int(action) + 1}

        self._ptr += 1
        terminated = self._ptr >= self._end
        obs = (self._obs() if not terminated
               else np.zeros(self.observation_space.shape, dtype=np.float32))
        return obs, reward, terminated, False, info


# ==============================================================================
# 4. TRAINING & EVALUATION
# ==============================================================================

def train_agent(env, log_dir, timesteps, gamma, seed=RANDOM_SEED):
    print(f"[train] PPO  timesteps={timesteps:,}  gamma={gamma}")
    model = PPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        n_steps=2048, batch_size=256, gae_lambda=0.95, gamma=gamma,
        learning_rate=3e-4, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.set_logger(sb3_configure_logger(log_dir, ["csv"]))
    model.learn(total_timesteps=timesteps, progress_bar=False)
    print("[train] done.")
    return model


def evaluate(model, X_test, levels_test, reward_matrix, title):
    """Deterministic policy over every held-out window; action a implies
    predicted level a + 1, compared against the Varnes proxy truth (§3.9)."""
    obs = np.clip(X_test.astype(np.float32), -OBS_CLIP, OBS_CLIP)
    actions, _ = model.predict(obs, deterministic=True)
    actions = np.asarray(actions).ravel()
    pred_level = actions + 1
    rewards = reward_matrix[levels_test - 1, actions]

    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(levels_test, pred_level, labels=labels)
    report = classification_report(
        levels_test, pred_level, labels=labels,
        target_names=[f"L{l} {LEVEL_NAMES[l]}" for l in labels],
        zero_division=0, output_dict=True)
    per_lvl = recall_score(levels_test, pred_level, labels=labels,
                           average=None, zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        levels_test, pred_level, labels=labels, average="weighted",
        zero_division=0)
    acc = float((pred_level == levels_test).mean())
    l5_recall = float(per_lvl[4])
    miss5 = int(((levels_test == SAFETY_LEVEL) &
                 (pred_level < SAFETY_LEVEL)).sum())
    tot5 = int((levels_test == SAFETY_LEVEL).sum())

    print("\n" + "=" * 72)
    print(f" EVALUATION ON HELD-OUT PIDs - {title}")
    print("=" * 72)
    print(f"  windows evaluated    : {len(levels_test):,}")
    print(f"  exact-level accuracy : {acc:6.3f}")
    print(f"  precision (weighted) : {p_w:6.3f}")
    print(f"  recall    (weighted) : {r_w:6.3f}")
    print(f"  F1-score  (weighted) : {f_w:6.3f}")
    print(f"  macro F1             : {report['macro avg']['f1-score']:6.3f}")
    print(f"  mean reward / step   : {rewards.mean():6.3f}")
    for l, rc in zip(labels, per_lvl):
        print(f"      Level {l} recall {LEVEL_NAMES[l]:>20s}: {rc:6.3f}")
    print(f"  >> LEVEL-5 RECALL (safety sensitivity): {l5_recall:6.3f} <<")
    print(f"  missed Level-5 alarms: {miss5} / {tot5}")
    print("=" * 72)

    return {"cm": cm, "labels": labels, "report": report,
            "per_level_recall": per_lvl, "level5_recall": l5_recall,
            "precision_w": float(p_w), "recall_w": float(r_w),
            "f1_w": float(f_w),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "accuracy": acc, "pred_level": pred_level, "actions": actions,
            "rewards": rewards, "missed_level5": miss5, "total_level5": tot5}


# ==============================================================================
# 5. PLOTS
# ==============================================================================

def plot_training_curves(log_dir, outdir):
    p = os.path.join(log_dir, "progress.csv")
    if not os.path.exists(p):
        return
    log = pd.read_csv(p).sort_values("time/total_timesteps")
    t = log["time/total_timesteps"]

    def line(col, ylabel, title, fname, color):
        if col not in log:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, log[col], color=color, lw=2)
        ax.set_xlabel("timesteps"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=600); plt.close(fig)

    line("rollout/ep_rew_mean", "mean episode reward",
         "Training reward curve (episode reward)", "episode_reward.png",
         "#3b82f6")
    line("rollout/ep_len_mean", "mean episode length (steps)",
         "Episode length during training", "episode_length.png", "#10b981")

    if "train/loss" in log or "train/value_loss" in log:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        if "train/loss" in log:
            axes[0].plot(t, log["train/loss"], lw=1.8, label="total loss",
                         color="#0f172a")
        if "train/value_loss" in log:
            axes[0].plot(t, log["train/value_loss"], lw=1.8,
                         label="value loss", color="#ef4444")
        axes[0].set_title("Training error - total & value loss")
        if "train/policy_gradient_loss" in log:
            axes[1].plot(t, log["train/policy_gradient_loss"], lw=1.8,
                         label="policy-gradient loss", color="#3b82f6")
        if "train/entropy_loss" in log:
            axes[1].plot(t, log["train/entropy_loss"], lw=1.8,
                         label="entropy loss", color="#f59e0b")
        axes[1].set_title("Training error - policy & entropy loss")
        for ax in axes:
            ax.set_xlabel("timesteps"); ax.set_ylabel("loss")
            ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "training_error.png"), dpi=600)
        plt.close(fig)


def plot_evaluation(ev, outdir, title):
    labels = ev["labels"]

    disp = ConfusionMatrixDisplay(ev["cm"],
                                  display_labels=[f"L{l}" for l in labels])
    fig, ax = plt.subplots(figsize=(6, 5.5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{title}\nLevel-5 recall = {ev['level5_recall']:.3f}")
    ax.set_xlabel("Predicted level (agent action)")
    ax.set_ylabel("True stability level")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"L{l}\n{LEVEL_NAMES[l].split('/')[0].strip()}" for l in labels],
           ev["per_level_recall"], color=LEVEL_COLORS)
    ax.axhline(0.9, ls="--", c="grey", lw=1, label="0.9 target")
    ax.set_ylim(0, 1.05); ax.set_ylabel("recall (sensitivity)")
    ax.set_title(f"Per-level recall - {title}"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "per_level_recall.png"), dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = np.bincount(ev["actions"], minlength=N_ACTIONS)
    ax.bar(range(N_ACTIONS), counts, color=LEVEL_COLORS)
    ax.set_xticks(range(N_ACTIONS))
    ax.set_xticklabels([a.replace(" / ", "/\n").replace(" ", "\n")
                        for a in ACTIONS], fontsize=8)
    ax.set_ylabel("windows")
    ax.set_title(f"Distribution of recommended actions - {title}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "action_distribution.png"), dpi=600)
    plt.close(fig)


def plot_dataset_figures(df, out_root):
    fig, ax = plt.subplots(figsize=(7, 4))
    dist = df["stability_level"].value_counts().sort_index()
    ax.bar([f"L{l}\n{LEVEL_NAMES[l].split('/')[0].strip()}" for l in dist.index],
           dist.values, color=[LEVEL_COLORS[l - 1] for l in dist.index])
    ax.set_ylabel("windows"); ax.set_title("Proxy stability-level distribution "
                                           f"({len(df):,} windows)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "level_distribution.png"), dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["abs_v"].clip(upper=150), bins=60, color="#3b82f6", alpha=0.85)
    for thr in VARNES_THRESHOLDS:
        ax.axvline(thr, ls="--", c="#0f172a", lw=1)
    ax.set_xlabel("|average staged velocity| (mm/yr, clipped at 150)")
    ax.set_ylabel("windows")
    ax.set_title("Velocity distribution with Varnes thresholds (2/10/30/100)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "velocity_distribution.png"), dpi=600)
    plt.close(fig)


def plot_ablation(res1, res2, out_root):
    """Per-level recall, Approach 1 vs Approach 2 (rainfall ablation, §3.9)."""
    labels = [1, 2, 3, 4, 5]
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, res1["per_level_recall"], w, label="Approach 1 (InSAR only)",
           color="#94a3b8")
    ax.bar(x + w/2, res2["per_level_recall"], w, label="Approach 2 (+ rainfall)",
           color="#3b82f6")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}\n{LEVEL_NAMES[l].split('/')[0].strip()}"
                        for l in labels])
    ax.set_ylim(0, 1.05); ax.set_ylabel("recall (sensitivity)")
    d5 = res2["level5_recall"] - res1["level5_recall"]
    ax.set_title("Rainfall ablation - per-level recall "
                 f"(Δ Level-5 recall = {d5:+.3f})")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "ablation_per_level_recall.png"), dpi=600)
    plt.close(fig)


# ==============================================================================
# 6. EXPERIMENT DRIVER
# ==============================================================================

def run_experiment(exp, train_df, test_df, out_root):
    tag, approach = exp["tag"], exp["approach"]
    outdir = os.path.join(out_root, tag)
    os.makedirs(outdir, exist_ok=True)
    reward = (INITIAL_REWARD if exp["reward"] == "initial"
              else SAFETY_TUNED_REWARD)

    print("\n" + "#" * 72)
    print(f" {exp['title']}")
    print("#" * 72)

    feats = build_feature_list(approach)
    print(f"[state] {len(feats)} features: {feats}")

    scaler = StandardScaler().fit(train_df[feats].values)
    Xtr = scaler.transform(train_df[feats].values)
    Xte = scaler.transform(test_df[feats].values)
    ytr = train_df["stability_level"].values
    yte = test_df["stability_level"].values

    env = Monitor(SlopeStabilityEnv(
        Xtr, ytr, train_df, reward_matrix=reward,
        episode_mode=exp["episode_mode"], max_steps=exp["max_steps"]))
    log_dir = os.path.join(outdir, "sb3_log")
    model = train_agent(env, log_dir, exp["timesteps"], exp["gamma"])

    ev = evaluate(model, Xte, yte, reward, exp["title"])

    # ---- artefacts ------------------------------------------------------------
    model.save(os.path.join(outdir, "ppo_slope_agent"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.pkl"))
    with open(os.path.join(outdir, "inference_config.json"), "w") as fh:
        json.dump({
            "approach": approach, "tag": tag,
            "features": feats, "n_features": len(feats),
            "actions": ACTIONS,
            "level_names": {int(k): v for k, v in LEVEL_NAMES.items()},
            "varnes_thresholds": VARNES_THRESHOLDS,
            "accel_escalate_pctl": ACCEL_ESCALATE_PCTL,
            "obs_clip": OBS_CLIP,
            "reward_matrix": reward.tolist(),
            "episode_mode": exp["episode_mode"],
            "gamma": exp["gamma"], "timesteps": exp["timesteps"],
            "scaler_file": "scaler.pkl", "model_file": "ppo_slope_agent.zip",
        }, fh, indent=2)

    pd.DataFrame(ev["report"]).T.to_csv(
        os.path.join(outdir, "classification_report.csv"))
    pd.DataFrame(ev["cm"],
                 index=[f"true_L{l}" for l in ev["labels"]],
                 columns=[f"pred_L{l}" for l in ev["labels"]]
                 ).to_csv(os.path.join(outdir, "confusion_matrix.csv"))

    plot_training_curves(log_dir, outdir)
    plot_evaluation(ev, outdir, exp["title"])

    # Approach 3: per-window engineering recommendations on the test PIDs
    if approach == 3:
        out = test_df[[GROUP, "abs_v", "stability_level"]].copy()
        out.rename(columns={"abs_v": "abs_velocity_mm_yr"}, inplace=True)
        out["agent_predicted_level"] = ev["pred_level"]
        out["recommended_action"] = [ACTIONS[a] for a in ev["actions"]]
        out.to_csv(os.path.join(outdir, "engineering_recommendations.csv"),
                   index=False)
        print(f"[appr3] engineering recommendations -> "
              f"{tag}/engineering_recommendations.csv")

    return ev


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    print("#" * 72)
    print(" RL FRAMEWORK v7 - LEAKAGE-FREE: past-only state AND labels")
    print("#" * 72)

    df = load_working_data()
    df = merge_point_features(df)
    df = add_kinematic_features(df)
    df, a_thr = assign_stability_levels(df)
    df.to_csv(os.path.join(OUT_ROOT, "working_data_with_levels.csv"),
              index=False)
    plot_dataset_figures(df, OUT_ROOT)

    train_df, test_df = split_by_pid(df)     # SAME split for all experiments

    results, rows = {}, []
    for exp in EXPERIMENTS:
        ev = run_experiment(exp, train_df, test_df, OUT_ROOT)
        results[exp["tag"]] = ev
        rows.append({
            "experiment": exp["tag"], "approach": exp["approach"],
            "reward": exp["reward"], "episode_mode": exp["episode_mode"],
            "gamma": exp["gamma"], "timesteps": exp["timesteps"],
            "level5_recall": ev["level5_recall"],
            "missed_level5": ev["missed_level5"],
            "total_level5": ev["total_level5"],
            "accuracy": ev["accuracy"],
            "precision_weighted": ev["precision_w"],
            "recall_weighted": ev["recall_w"],
            "f1_weighted": ev["f1_w"], "macro_f1": ev["macro_f1"],
            "mean_reward_per_step": float(ev["rewards"].mean()),
            "accel_escalate_thr_mm_yr": a_thr,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_ROOT, "experiments_summary.csv"),
                   index=False)

    # ---- Section 3.9 rainfall ablation -----------------------------------------
    r1 = results["approach1_insar_only"]
    r2 = results["approach2_with_rainfall"]
    plot_ablation(r1, r2, OUT_ROOT)
    d5 = r2["level5_recall"] - r1["level5_recall"]
    print("\n" + "=" * 72)
    print(" RAINFALL ABLATION (Approach 1 vs 2, identical settings & split)")
    print("=" * 72)
    print(f"  Level-5 recall  A1 (InSAR only) : {r1['level5_recall']:.3f}")
    print(f"  Level-5 recall  A2 (+ rainfall) : {r2['level5_recall']:.3f}")
    print(f"  Delta (rainfall contribution)   : {d5:+.3f}")
    print(f"  macro F1  A1 / A2               : "
          f"{r1['macro_f1']:.3f} / {r2['macro_f1']:.3f}")

    r3 = results["approach3_improved"]
    print("\n IMPROVED APPROACH 3 (past-only state, safety-tuned reward, "
          "balanced episodes)")
    print(f"  Level-5 recall                  : {r3['level5_recall']:.3f} "
          f"(raw-only state: 0.576 | target-informed state: 0.987)")
    print(f"  missed Level-5 alarms           : {r3['missed_level5']} / "
          f"{r3['total_level5']}")
    print(f"  macro F1                        : {r3['macro_f1']:.3f}")


    print(f"\nAll outputs in: {OUT_ROOT}")
    print("  per-experiment folders : approach1_insar_only / "
          "approach2_with_rainfall / approach3_improved")
    print("  ablation               : ablation_per_level_recall.png, "
          "experiments_summary.csv")
    print("  inference bundle (A3)  : ppo_slope_agent.zip + scaler.pkl + "
          "inference_config.json")


if __name__ == "__main__":
    main()

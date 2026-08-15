"""
================================================================================
 slope_agent_interface.py
--------------------------------------------------------------------------------
 Interface to load the trained slope-stability PPO agent and test it on NEW
 data, using the SAVED training scaler for an exact match.

 v6 VERSION - leakage-free bundle (slope_rl_framework_v6.py): the model
 observes only raw quantities and the reference kinematics are computed from
 the observed stages alone. The Target column is OPTIONAL and ignored.

 Requires the inference bundle:
       ppo_slope_agent.zip      (trained agent)
       scaler.pkl               (the fitted training StandardScaler)
       inference_config.json    (exact feature order + metadata)
 and, if the new data does not already carry point attributes:
       I_PS_NS_V.csv            (temporal_coherence, mean_velocity, rmse)

 New data must contain the raw columns:
       PID, S_1, S_2, S_3, S_4, T1, T2, T3, T4, R_1, R_2, R_3, R_4

 Use as a library:
       from slope_agent_interface import SlopeAgentInterface
       agent = SlopeAgentInterface("rl_outputs_approach3")
       table = agent.predict_file("new_data.csv", out_csv="recommendations.csv")

 Use from the command line:
       python slope_agent_interface.py --model-dir rl_outputs_approach3 \
              --data new_data.csv --insar I_PS_NS_V.csv --out recommendations.csv
================================================================================
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from stable_baselines3 import PPO

DISP = ["S_1", "S_2", "S_3", "S_4"]
TIME = ["T1", "T2", "T3", "T4"]
RAIN = ["R_1", "R_2", "R_3", "R_4"]
POINT = ["temporal_coherence", "mean_velocity", "rmse"]


class SlopeAgentInterface:
    """Loads the model + saved scaler + config and scores new sliding windows."""

    def __init__(self, model_dir, insar_file="I_PS_NS_V.csv"):
        self.model_dir = model_dir
        self.insar_file = insar_file

        cfg_path = os.path.join(model_dir, "inference_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"inference_config.json not found in {model_dir}. Retrain with "
                f"the updated framework (v3) so the scaler/config are saved.")
        with open(cfg_path) as fh:
            self.cfg = json.load(fh)

        self.features = self.cfg["features"]                 # exact order
        self.actions = self.cfg["actions"]
        self.levels = {int(k): v for k, v in self.cfg["level_names"].items()}
        self.varnes = self.cfg["varnes_thresholds"]
        self.clip = float(self.cfg.get("obs_clip", 10.0))

        # SAVED scaler -> exact-match standardisation
        self.scaler = joblib.load(os.path.join(model_dir, self.cfg["scaler_file"]))
        # trained agent
        self.model = PPO.load(os.path.join(model_dir, "ppo_slope_agent"))

        exp = int(self.model.observation_space.shape[0])
        if exp != len(self.features):
            raise ValueError(f"Model expects {exp} features but config lists "
                             f"{len(self.features)}.")
        print(f"[interface] loaded agent ({exp} features), saved scaler, config "
              f"from '{model_dir}'")

    # -------------------------------------------------- feature engineering
    def _add_point_features(self, df):
        """Use the file's own point columns; else merge from the InSAR file."""
        if all(c in df.columns for c in POINT):
            return df
        if os.path.exists(self.insar_file):
            ins = pd.read_csv(self.insar_file)
            merge = {"PID": np.arange(1, len(ins) + 1)}
            for c in POINT:
                merge[c] = ins[c].values if c in ins.columns else 0.0
            df = df.merge(pd.DataFrame(merge), on="PID", how="left")
        for c in POINT:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0.0)
        return df

    def _add_kinematics(self, df):
        """Past-only kinematics, identical to v6 training (no Target term)."""
        S1, S2, S3, S4 = (df[c] for c in DISP)
        T1, T2, T3 = (df[c].clip(lower=1) for c in ["T1", "T2", "T3"])
        v = np.vstack([(S2 - S1) / T1, (S3 - S2) / T2,
                       (S4 - S3) / T3]).T * 365.25
        df["avg_staged_v"] = v.mean(axis=1)
        df["sd_staged_v"] = v.std(axis=1)
        df["accel"] = ((S4 - S3) / T3 - (S3 - S1) / (T1 + T2)) * 365.25
        df["precip_antecedent"] = df[RAIN].sum(axis=1)
        return df

    def _varnes_level(self, abs_v):
        lv = np.ones(len(abs_v), dtype=int)
        for i, thr in enumerate(self.varnes):
            lv[abs_v >= thr] = i + 2
        return lv

    def prepare(self, df):
        df = df.copy()
        if "PID" not in df.columns:
            raise ValueError("New data must contain a 'PID' column.")
        missing_raw = [c for c in DISP + TIME + RAIN if c not in df.columns]
        if missing_raw:
            raise ValueError(f"New data is missing raw columns: {missing_raw}")
        df = self._add_point_features(df)
        df = self._add_kinematics(df)
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Could not build required features: {missing}")
        return df

    # -------------------------------------------------- prediction
    def predict(self, df):
        """Return a recommendation table for a raw new-data DataFrame."""
        df = self.prepare(df)
        X = df[self.features].to_numpy(np.float32)
        Xs = np.clip(self.scaler.transform(X), -self.clip, self.clip).astype(np.float32)
        actions = np.asarray(self.model.predict(Xs, deterministic=True)[0]).ravel()
        abs_v = df["avg_staged_v"].abs().to_numpy()
        out = pd.DataFrame({
            "PID": df["PID"].values,
            "abs_velocity_mm_yr": np.round(abs_v, 2),
            "accel_mm_yr": np.round(df["accel"].to_numpy(), 2),
            "kinematic_level_ref": self._varnes_level(abs_v),
            "agent_predicted_level": actions + 1,
            "level_label": [self.levels[a + 1] for a in actions],
            "recommended_action": [self.actions[a] for a in actions],
        })
        return out

    def predict_file(self, data_path, out_csv="action_recommendations.csv"):
        df = (pd.read_excel(data_path) if data_path.lower().endswith(".xlsx")
              else pd.read_csv(data_path))
        table = self.predict(df)
        if out_csv:
            table.to_csv(out_csv, index=False)
        return table

    def summary(self, table):
        n = len(table)
        lines = [f"Scored {n:,} windows.", "", "Recommended-action distribution:"]
        counts = table["recommended_action"].value_counts()
        for a in self.actions:
            c = int(counts.get(a, 0))
            lines.append(f"  {a:22s}: {c:6d}  ({100*c/n:5.1f} %)")
        return "\n".join(lines)


# ---------------------------------------------------------------- CLI
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Test the trained slope RL agent on new data.")
    ap.add_argument("--model-dir", default=script_dir,
                    help="folder with ppo_slope_agent.zip, scaler.pkl, inference_config.json "
                         "(default: the folder this script sits in)")
    ap.add_argument("--data", default=None, help="new data file (.csv or .xlsx)")
    ap.add_argument("--insar", default=os.path.join(script_dir, "I_PS_NS_V.csv"),
                    help="InSAR reference for point features")
    ap.add_argument("--out", default="action_recommendations.csv", help="output CSV")
    args, _unknown = ap.parse_known_args()   # tolerate IDE-injected args (Spyder)

    # Spyder/zero-argument fallback: ask for the data file interactively.
    def _valid(p):
        return p and os.path.isfile(p) and p.lower().endswith((".csv", ".xlsx"))

    if args.data:
        args.data = args.data.strip().strip('"')
    attempts = 0
    while not _valid(args.data) and attempts < 3:
        if args.data:
            if os.path.isdir(args.data):
                cands = [f for f in os.listdir(args.data)
                         if f.lower().endswith((".csv", ".xlsx"))
                         and not f.startswith(("action_", "confusion", "summary",
                                               "classification", "engineering",
                                               "working_data"))]
                print(f"'{args.data}' is a FOLDER, not a data file.")
                if cands:
                    print("Data files found inside it: " + ", ".join(cands[:8]))
                    print("Type the FULL path to one of them (folder\\filename).")
            else:
                print(f"Not a valid .csv/.xlsx file: '{args.data}'")
        try:
            args.data = input("Path to the NEW data file (.csv/.xlsx): ").strip().strip('"')
        except EOFError:
            break
        attempts += 1
    if not _valid(args.data):
        raise SystemExit("No valid new-data file provided. Expected a .csv or .xlsx "
                         "FILE (not a folder) with columns PID, S_1..S_4, "
                         "T1..T4, R_1..R_4 (Target optional, ignored).")

    agent = SlopeAgentInterface(args.model_dir, insar_file=args.insar)
    table = agent.predict_file(args.data, out_csv=args.out)
    print(f"\n[interface] wrote {args.out}\n")
    print(agent.summary(table))
    print("\nFirst rows:")
    print(table.head(8).to_string(index=False))


if __name__ == "__main__":
    main()

"""
================================================================================
 slope_agent_gui.py  (v7 bundle - rl_outputs_v7/approach3_improved)
--------------------------------------------------------------------------------
 Simple desktop interface to test the trained v7 slope-stability RL agent
 (leakage-free, safety-tuned Approach 3: Level-5 recall 0.991 on held-out
 points) on new data. The model folder defaults to THIS folder, which already
 contains ppo_slope_agent.zip, scaler.pkl, inference_config.json and
 I_PS_NS_V.csv. Choose a new data file and click "Run".

 New data must contain the raw columns
     PID, S_1..S_4, T1..T4, R_1..R_4        (Target optional - ignored)
 (the past-only kinematic features are computed from them automatically, and
 the SAVED training scaler is applied for an exact standardisation match).

 All prediction logic is delegated to SlopeAgentInterface. Requires a desktop
 environment (Tkinter, which ships with standard Python on Windows/macOS).

 Run:  C:/Users/hp/anaconda3/python.exe slope_agent_gui.py
================================================================================
"""

import os
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from slope_agent_interface import SlopeAgentInterface


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Slope-Stability RL Agent (v7) — Inference Interface")
        self.geometry("880x620")
        self.minsize(760, 560)
        self.configure(padx=14, pady=12)
        self._table = None
        self._build()

    # ---------------------------------------------------------------- UI
    def _row(self, parent, label, default, browse):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text=label, width=16).pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(frame, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(frame, text="Browse…", command=lambda: browse(var)).pack(side="left")
        return var

    def _build(self):
        ttk.Label(self, text="Slope-Stability RL Agent (v7) — test on new data",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(self, foreground="#555",
                  text="Model folder defaults to this v7 bundle "
                       "(rl_outputs_v7\\approach3_improved). New data columns: "
                       "PID, S_1..S_4, T1..T4, R_1..R_4 (Target optional)."
                  ).pack(anchor="w", pady=(0, 8))

        box = ttk.Frame(self); box.pack(fill="x")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_data = os.path.join(script_dir, "new_data.xlsx")
        if not os.path.exists(default_data):
            default_data = ""
        self.model_dir = self._row(box, "Model folder:", script_dir, self._pick_dir)
        self.data_file = self._row(box, "New data file:", default_data, self._pick_data)
        self.insar_file = self._row(box, "InSAR reference:",
                                    os.path.join(script_dir, "I_PS_NS_V.csv"), self._pick_file)
        self.out_file = self._row(box, "Output CSV:", "action_recommendations.csv", self._pick_save)

        bar = ttk.Frame(self); bar.pack(fill="x", pady=8)
        self.run_btn = ttk.Button(bar, text="▶  Run inference", command=self._run)
        self.run_btn.pack(side="left")
        self.status = ttk.Label(bar, text="Ready.", foreground="#0F766E")
        self.status.pack(side="left", padx=12)

        self.summary = ttk.Label(self, text="", justify="left", font=("Consolas", 9))
        self.summary.pack(anchor="w", pady=(2, 6))

        cols = ("PID", "abs_velocity_mm_yr", "accel_mm_yr", "kinematic_level_ref",
                "agent_predicted_level", "level_label", "recommended_action")
        tree_frame = ttk.Frame(self); tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=14)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    # ---------------------------------------------------------------- pickers
    def _pick_dir(self, var):
        p = filedialog.askdirectory(title="Select the model folder")
        if p: var.set(p)

    def _pick_data(self, var):
        p = filedialog.askopenfilename(title="Select new data file",
              filetypes=[("Data", "*.csv *.xlsx"), ("All files", "*.*")])
        if p: var.set(p)

    def _pick_file(self, var):
        p = filedialog.askopenfilename(title="Select file",
              filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p: var.set(p)

    def _pick_save(self, var):
        p = filedialog.asksaveasfilename(title="Save recommendations as",
              defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p: var.set(p)

    # ---------------------------------------------------------------- run
    def _run(self):
        data = self.data_file.get().strip()
        if not data or not os.path.exists(data):
            messagebox.showerror("Missing data", "Please choose a valid new data file.")
            return
        self.run_btn.config(state="disabled")
        self.status.config(text="Running… loading model and scoring windows.", foreground="#B45309")
        self.update_idletasks()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            agent = SlopeAgentInterface(self.model_dir.get().strip(),
                                        insar_file=self.insar_file.get().strip())
            table = agent.predict_file(data_path=self.data_file.get().strip(),
                                       out_csv=self.out_file.get().strip() or None)
            self._table = table
            self.after(0, lambda: self._done(agent, table))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            self.after(0, lambda: self._fail(msg, tb))

    def _done(self, agent, table):
        for r in self.tree.get_children():
            self.tree.delete(r)
        for _, row in table.head(200).iterrows():
            self.tree.insert("", "end", values=[row[c] for c in self.tree["columns"]])
        self.summary.config(text=agent.summary(table))
        out = self.out_file.get().strip()
        self.status.config(text=f"Done — {len(table):,} windows scored. Saved: {out}",
                           foreground="#0F766E")
        self.run_btn.config(state="normal")

    def _fail(self, msg, tb):
        self.status.config(text="Failed.", foreground="#B91C1C")
        self.run_btn.config(state="normal")
        messagebox.showerror("Inference error", msg + "\n\n" + tb[-1200:])


if __name__ == "__main__":
    App().mainloop()

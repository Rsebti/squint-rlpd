#!/usr/bin/env python3
"""Re-fit FOPDT model on existing step-response CSVs (offline, no robot).

Picks the most recent CSV for each ARM_JOINT, refits with scipy now that
it's available, and prints the same summary table as probe_all_joints.py.
"""
from __future__ import annotations

import csv as _csv
import glob
import math
import sys
from pathlib import Path

import numpy as np

# Make root-level probe_*.py importable when this lives in debug_scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_step_response import fit_fopdt, DT_CTRL_DEFAULT, ARTIFACT_DIR

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex",   "wrist_roll")


def load_csv(path: Path):
    with path.open() as f:
        rows = list(_csv.reader(f))[1:]
    t = np.array([float(r[0]) for r in rows])
    lab = [r[1] for r in rows]
    y = np.array([float(r[2]) for r in rows])
    return t, lab, y


def main():
    dt_ctrl = DT_CTRL_DEFAULT
    print("=" * 78)
    print(f"{'joint':<14} {'L ms':>7} {'tau ms':>8} {'R^2':>6} "
          f"{'delay_steps':>12} {'lag_alpha':>10}")
    print("-" * 78)
    Ls, taus = [], []
    for j in ARM_JOINTS:
        files = sorted(ARTIFACT_DIR.glob(f"step_response_{j}_*.csv"))
        if not files:
            print(f"{j:<14}  no CSV found")
            continue
        path = files[-1]
        t, lab, y = load_csv(path)
        # Find first "step" label timestamp
        i_step = next((i for i, l in enumerate(lab) if l == "step"), None)
        if i_step is None:
            print(f"{j:<14}  no 'step' segment in CSV {path.name}")
            continue
        t_step = t[i_step]
        # Use baseline + step segments only; drop the return segment so the
        # fit isn't confused by the recovery.
        i_ret = next((i for i, l in enumerate(lab) if l == "return"), len(lab))
        t_fit = t[:i_ret]
        y_fit = y[:i_ret]
        try:
            fit = fit_fopdt(t_fit, y_fit, t_step)
        except Exception as e:
            print(f"{j:<14}  fit failed: {e}")
            continue
        L_ms   = fit["L"] * 1000.0
        tau_ms = fit["tau"] * 1000.0
        n      = int(round(fit["L"] / dt_ctrl))
        alpha  = dt_ctrl / (dt_ctrl + max(fit["tau"], 1e-6))
        print(f"{j:<14} {L_ms:>7.1f} {tau_ms:>8.1f} {fit['r2']:>6.3f} "
              f"{n:>12d} {alpha:>10.3f}")
        Ls.append(fit["L"]); taus.append(fit["tau"])
    if Ls:
        L_mean   = float(np.mean(Ls))
        tau_mean = float(np.mean(taus))
        print("-" * 78)
        print(f"mean over {len(Ls)} joints:  "
              f"L={L_mean*1000:.1f} ms   tau={tau_mean*1000:.1f} ms")
        print(f"  -> ACTION_DELAY_STEPS_DEFAULT = "
              f"{int(round(L_mean/dt_ctrl))}")
        print(f"  -> LAG_ALPHA_DEFAULT          = "
              f"{dt_ctrl/(dt_ctrl+tau_mean):.3f}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())

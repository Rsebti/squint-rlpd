# Same 6-cube calibration row as sweep_cube_brightness.py, but using the
# Friday-2026-05-15 commit 2783d83 palette so we can compare side-by-side
# before committing to a revert.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Monkey-patch the palette BEFORE importing the existing sweep so that
# COLOR_PALETTE-references inside the env resolve to the Friday values.
import numpy as np
import envs  # noqa: F401  — registers SO101 envs
import envs.place as place_mod

FRIDAY_PALETTE = np.array(
    [
        [187/255,  47/255,  27/255],  # 0 red
        [  6/255,  33/255, 111/255],  # 1 blue
        [ 24/255,  72/255,  30/255],  # 2 green
        [216/255, 195/255,  73/255],  # 3 yellow
        [ 80/255,  43/255,  82/255],  # 4 purple
        [216/255,  86/255,  54/255],  # 5 orange
    ],
    dtype=np.float32,
)
place_mod.COLOR_PALETTE = FRIDAY_PALETTE
place_mod.NUM_COLORS = len(FRIDAY_PALETTE)

# Now import the sweep module — its `_set_palette_color_with_scale` reads
# `place_mod.COLOR_PALETTE` at call time, so the override propagates.
import sweep_cube_brightness as scb  # noqa: E402

# Redirect output to a different file so we don't overwrite the HEAD-palette
# sweep result.
def _save_under_friday_name():
    import cv2
    values = [0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.55, 1.80]
    panels = []
    for i, v in enumerate(values):
        print(f"[{i+1}/{len(values)}] FRIDAY palette, brightness_scale={v:.2f}", flush=True)
        env = scb.make_env(n_distractors=5)
        scb.lay_out_six_cubes(env, brightness_scale=v)
        render = scb.capture(env)
        env.close()
        cropped = scb.crop_to_cube_row(render)
        target_w = 800
        scale = target_w / cropped.shape[1]
        target_h = int(round(cropped.shape[0] * scale))
        big = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        big_bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
        panels.append(scb.label_panel(big_bgr,
            f"#{i+1}  FRIDAY palette, v_scale={v:.2f}  (R B G Y P O)"))

    cols = 2
    rows = [np.hstack(panels[r*cols:(r+1)*cols]) for r in range(len(panels)//cols)]
    full = np.vstack(rows)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cube_brightness_friday.png")
    cv2.imwrite(out_path, full)
    print(f"\nSaved: {out_path}  ({full.shape[1]}w × {full.shape[0]}h)")
    print(f"Values (dark→bright): {values}")
    print("Palette = Friday 2026-05-15 commit 2783d83.")


if __name__ == "__main__":
    _save_under_friday_name()

# Specular sweep on the cube material. Specular is the white reflection layer
# PBR adds on top of base_color — high values look like "color mixed with
# white". Roughness=0.825 (matte) is held constant so we're only varying the
# strength of the white sheen.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import colorsys
import cv2
import envs  # noqa: F401
import envs.place as place_mod


FRIDAY = np.array(
    [
        [187/255,  47/255,  27/255],
        [  6/255,  33/255, 111/255],
        [ 24/255,  72/255,  30/255],
        [216/255, 195/255,  73/255],
        [ 80/255,  43/255,  82/255],
        [216/255,  86/255,  54/255],
    ],
    dtype=np.float32,
)
NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]
HUE_OVERRIDE_DEG = {"orange": 28.0}
TARGET_V = {
    "red": 0.50, "blue": 0.55, "green": 0.45, "yellow": 0.85,
    "purple": 0.35, "orange": 0.65,
}
SAT_LIFT = 0.50  # palette anchor user said looks good


def make_palette(sat_lift: float = SAT_LIFT) -> np.ndarray:
    out = []
    for name, rgb in zip(NAMES, FRIDAY):
        h, s, v = colorsys.rgb_to_hsv(*rgb.tolist())
        if name in HUE_OVERRIDE_DEG:
            h = (HUE_OVERRIDE_DEG[name] % 360) / 360.0
        s2 = s + (1.0 - s) * sat_lift
        v2 = TARGET_V[name]
        out.append(list(colorsys.hsv_to_rgb(h, s2, v2)))
    return np.array(out, dtype=np.float32)


# Fix the palette before importing scb so anything that captures it at import
# time sees the right values.
place_mod.COLOR_PALETTE = make_palette()
place_mod.NUM_COLORS = len(place_mod.COLOR_PALETTE)


import sweep_cube_brightness as scb  # noqa: E402


def render_with_specular(spec: float):
    # `_set_palette_color_with_scale` reads from `domain_randomization_config
    # .item_specular_range` at call time, so injecting via PINNED before
    # make_env is enough.
    scb.PINNED["item_specular_range"] = (spec, spec)
    env = scb.make_env(n_distractors=5)
    scb.lay_out_six_cubes(env, brightness_scale=1.0)
    render = scb.capture(env)
    env.close()
    return render


def main():
    values = [0.00, 0.005, 0.01, 0.02, 0.05, 0.08, 0.12, 0.20, 0.30, 0.45]
    panels = []

    for i, spec in enumerate(values):
        print(f"[{i+1}/{len(values)}] item_specular={spec:.3f}", flush=True)
        render = render_with_specular(spec)

        cropped = scb.crop_to_cube_row(render)
        target_w = 800
        scale = target_w / cropped.shape[1]
        target_h = int(round(cropped.shape[0] * scale))
        big = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        big_bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
        marker = " (current default 0.05)" if abs(spec - 0.05) < 1e-6 else ""
        panels.append(scb.label_panel(big_bgr,
            f"#{i+1}  item_specular={spec:.3f}{marker}"))

    cols = 2
    rows = [np.hstack(panels[r*cols:(r+1)*cols]) for r in range(len(panels)//cols)]
    full = np.vstack(rows)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cube_specular.png")
    cv2.imwrite(out_path, full)
    print(f"\nSaved: {out_path}  ({full.shape[1]}w × {full.shape[0]}h)")
    print(f"Palette: sat_lift={SAT_LIFT}, per-color V, orange@28°.")
    print("Lower specular = less white-sheen mixed in.")


if __name__ == "__main__":
    main()

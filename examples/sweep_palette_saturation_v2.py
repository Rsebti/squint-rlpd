# Saturation sweep with PER-COLOR brightness targets. The previous sweep used
# one uniform val_lift, which made red/orange/purple come out too bright in the
# render — they have low Friday V (0.32-0.85) but a multiplicative lift pushed
# them all together. Here every colour gets its own V target so the dark ones
# (red, orange, purple) stay dark while the originally-dark ones (blue, green)
# brighten enough to read.

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

# Hue overrides (None = keep Friday hue).
HUE_OVERRIDE_DEG = {
    "orange": 28.0,   # separate from red at H≈7.5° after S boost
}

# Per-color target V. Chosen to bring red/orange/purple down (user said
# they came out too bright in the previous render) while lifting the
# originally-dark blue and green enough to read against the table.
TARGET_V = {
    "red":    0.50,   # deep matte red
    "blue":   0.55,   # lifted from Friday 0.44
    "green":  0.45,   # lifted from Friday 0.28
    "yellow": 0.85,   # ≈ Friday
    "purple": 0.35,   # slightly above Friday 0.32
    "orange": 0.65,   # darker tangerine
}


def make_palette(sat_lift: float):
    """Lift each color's S by `sat_lift` toward 1.0, set V to TARGET_V[name],
    and optionally override H."""
    out = []
    for i, (name, rgb) in enumerate(zip(NAMES, FRIDAY)):
        h, s, v = colorsys.rgb_to_hsv(*rgb.tolist())
        if name in HUE_OVERRIDE_DEG:
            h = (HUE_OVERRIDE_DEG[name] % 360) / 360.0
        s2 = s + (1.0 - s) * sat_lift
        v2 = TARGET_V[name]
        out.append(list(colorsys.hsv_to_rgb(h, s2, v2)))
    return np.array(out, dtype=np.float32)


SAT_LIFTS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.78, 0.85, 0.92, 0.98]


import sweep_cube_brightness as scb  # noqa: E402


def render_with_palette(palette: np.ndarray):
    place_mod.COLOR_PALETTE = palette
    place_mod.NUM_COLORS = len(palette)
    env = scb.make_env(n_distractors=5)
    scb.lay_out_six_cubes(env, brightness_scale=1.0)
    render = scb.capture(env)
    env.close()
    return render


def main():
    panels = []
    palettes_for_log = []

    for i, s in enumerate(SAT_LIFTS):
        print(f"[{i+1}/{len(SAT_LIFTS)}] sat_lift={s:.2f}  per-color V targets",
              flush=True)
        palette = make_palette(s)
        palettes_for_log.append((s, palette))
        render = render_with_palette(palette)

        cropped = scb.crop_to_cube_row(render)
        target_w = 800
        scale = target_w / cropped.shape[1]
        target_h = int(round(cropped.shape[0] * scale))
        big = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        big_bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
        panels.append(scb.label_panel(big_bgr,
            f"#{i+1}  sat_lift={s:.2f}  (per-color V, orange@28°)"))

    cols = 2
    rows = [np.hstack(panels[r*cols:(r+1)*cols]) for r in range(len(panels)//cols)]
    full = np.vstack(rows)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "palette_saturation_v2.png")
    cv2.imwrite(out_path, full)
    print(f"\nSaved: {out_path}  ({full.shape[1]}w × {full.shape[0]}h)")

    print(f"\nPer-color V targets: {TARGET_V}")
    print(f"Hue overrides:       {HUE_OVERRIDE_DEG}")
    print("\n" + "=" * 70)
    for s, palette in palettes_for_log:
        print(f"\n# sat_lift={s:.2f}")
        for name, rgb in zip(NAMES, palette):
            r255 = int(round(rgb[0] * 255))
            g255 = int(round(rgb[1] * 255))
            b255 = int(round(rgb[2] * 255))
            print(f"#   {name:7} = ({r255:3}, {g255:3}, {b255:3})")


if __name__ == "__main__":
    main()

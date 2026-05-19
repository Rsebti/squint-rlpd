# Render the cube row with a "Friday + boost" palette: each Friday colour's
# saturation and brightness lifted halfway toward 1.0. The lift-toward-1.0
# rule preserves ordering (red and orange stay distinct, vs a multiplicative
# boost that capped both at S≈0.95).

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import colorsys
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

# `t` controls how aggressively to boost. S_new = S_old + (1 - S_old)*t.
# t=0 → Friday; t=1 → fully saturated / max-bright.
SAT_LIFT = 0.50
VAL_LIFT = 0.50


def boosted_palette():
    out = []
    for rgb in FRIDAY:
        h, s, v = colorsys.rgb_to_hsv(*rgb.tolist())
        s2 = s + (1.0 - s) * SAT_LIFT
        v2 = v + (1.0 - v) * VAL_LIFT
        r2, g2, b2 = colorsys.hsv_to_rgb(h, s2, v2)
        out.append([r2, g2, b2])
    return np.array(out, dtype=np.float32)


BOOSTED = boosted_palette()
place_mod.COLOR_PALETTE = BOOSTED
place_mod.NUM_COLORS = len(BOOSTED)


# Now import the sweep module (it reads place_mod.COLOR_PALETTE at call time).
import sweep_cube_brightness as scb  # noqa: E402
import cv2  # noqa: E402


def main():
    print("Boosted palette (RGB 0-255):")
    for name, rgb in zip(NAMES, BOOSTED):
        rgb255 = tuple(int(round(c * 255)) for c in rgb)
        print(f"  {name:7} = ({rgb255[0]:3}, {rgb255[1]:3}, {rgb255[2]:3})")

    env = scb.make_env(n_distractors=5)
    scb.lay_out_six_cubes(env, brightness_scale=1.0)
    render = scb.capture(env)
    env.close()

    cropped = scb.crop_to_cube_row(render)
    target_w = 1200
    scale = target_w / cropped.shape[1]
    target_h = int(round(cropped.shape[0] * scale))
    big = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    big_bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
    big_bgr = scb.label_panel(big_bgr,
        f"FRIDAY + (sat_lift={SAT_LIFT}, val_lift={VAL_LIFT})   R B G Y P O")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cube_brightness_friday_boosted.png")
    cv2.imwrite(out_path, big_bgr)
    print(f"\nSaved: {out_path}")

    # Also emit the paste-ready palette block.
    print("\nPaste-ready place.py palette block:")
    print("COLOR_PALETTE = np.array([")
    for name, rgb in zip(NAMES, BOOSTED):
        rgb255 = tuple(int(round(c * 255)) for c in rgb)
        print(f"    [{rgb255[0]:3}/255, {rgb255[1]:3}/255, {rgb255[2]:3}/255],  # {name}")
    print("], dtype=np.float32)")


if __name__ == "__main__":
    main()

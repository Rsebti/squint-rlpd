# Fine-tune orange hue toward red. Everything else stays at the just-baked
# palette + locked lighting. Sweep H from 10° (vermillion / red-orange) to
# 32° (current 28° + a slightly more yellow option, just for reference).

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import colorsys
import cv2
import envs  # noqa: F401
import envs.place as place_mod


# Current baked palette in place.py (post-edit).
BAKED = np.array(
    [
        [128/255,  18/255,   3/255],  # 0 red
        [  1/255,  37/255, 140/255],  # 1 blue
        [  6/255, 115/255,  19/255],  # 2 green
        [217/255, 187/255,  11/255],  # 3 yellow
        [ 85/255,   7/255,  89/255],  # 4 purple
        [166/255,  81/255,   6/255],  # 5 orange
    ],
    dtype=np.float32,
)
NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]


def palette_with_orange_hue(deg: float) -> np.ndarray:
    out = BAKED.copy()
    _, s, v = colorsys.rgb_to_hsv(*BAKED[5].tolist())
    r, g, b = colorsys.hsv_to_rgb((deg % 360) / 360.0, s, v)
    out[5] = [r, g, b]
    return out


import sweep_cube_brightness as scb  # noqa: E402


def render(palette: np.ndarray):
    place_mod.COLOR_PALETTE = palette
    place_mod.NUM_COLORS = len(palette)
    env = scb.make_env(n_distractors=5)
    scb.lay_out_six_cubes(env, brightness_scale=1.0)
    img = scb.capture(env)
    env.close()
    return img


def main():
    # 10 values, biased below current 28° toward red.
    hues = [10.0, 13.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 32.0]
    panels = []

    for i, h in enumerate(hues):
        print(f"[{i+1}/{len(hues)}] orange_hue={h:.0f}°", flush=True)
        palette = palette_with_orange_hue(h)
        img = render(palette)
        cropped = scb.crop_to_cube_row(img)
        target_w = 800
        scale = target_w / cropped.shape[1]
        target_h = int(round(cropped.shape[0] * scale))
        big = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        big_bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)

        # Compute & embed the resulting orange RGB so the user can pick by #.
        rgb255 = tuple(int(round(c * 255)) for c in palette[5])
        marker = "  ← current" if abs(h - 28.0) < 1e-6 else ""
        panels.append(scb.label_panel(big_bgr,
            f"#{i+1}  orange H={int(h)}°  RGB=({rgb255[0]},{rgb255[1]},{rgb255[2]}){marker}"))

    cols = 2
    rows = [np.hstack(panels[r*cols:(r+1)*cols]) for r in range(len(panels)//cols)]
    full = np.vstack(rows)

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "sweeps", "orange_hue.png")
    cv2.imwrite(out_path, full)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

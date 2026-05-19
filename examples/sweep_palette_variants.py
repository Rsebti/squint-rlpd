# Grid of 10 palette variants around the Friday-boosted point:
#   * 8 variants sweeping sat_lift × val_lift near (0.5, 0.5)
#   * 2 variants with orange hue nudged toward 28° so it visibly separates
#     from red after the saturation lift.
#
# All rendered with the SAME lock context: exposure=1.80, table_albedo=0.85,
# bowl_emission=0.80, no DR jitter on cubes, calibration camera in front of
# the row.

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


def make_palette(sat_lift: float, val_lift: float, orange_hue_deg: float | None = None):
    """Lift each Friday color's S and V halfway-toward-1.0 by the given amounts.
    Optionally override orange's hue (palette index 5) before the lift."""
    out = []
    for i, rgb in enumerate(FRIDAY):
        h, s, v = colorsys.rgb_to_hsv(*rgb.tolist())
        if i == 5 and orange_hue_deg is not None:
            h = (orange_hue_deg % 360) / 360.0
        s2 = s + (1.0 - s) * sat_lift
        v2 = v + (1.0 - v) * val_lift
        out.append(list(colorsys.hsv_to_rgb(h, s2, v2)))
    return np.array(out, dtype=np.float32)


# 10 variants — label, sat_lift, val_lift, orange_hue_deg
VARIANTS = [
    ("less boost",        0.35, 0.35, None),
    ("slight less",       0.45, 0.45, None),
    ("anchor",            0.50, 0.50, None),
    ("more sat",          0.65, 0.50, None),
    ("more bright",       0.50, 0.65, None),
    ("more both",         0.60, 0.60, None),
    ("high sat",          0.75, 0.55, None),
    ("very bright",       0.55, 0.75, None),
    ("anchor + ORANGE@28", 0.50, 0.50, 28.0),
    ("more both + ORANGE@28", 0.60, 0.60, 28.0),
]


# We need to swap COLOR_PALETTE per render. Because the sweep module captures
# the palette by reference at call time inside _set_palette_color_with_scale,
# reassigning place_mod.COLOR_PALETTE before each capture is sufficient.
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

    for i, (label, s, v, oh) in enumerate(VARIANTS):
        print(f"[{i+1}/{len(VARIANTS)}] {label}  "
              f"sat_lift={s:.2f} val_lift={v:.2f} orange_h={oh}", flush=True)
        palette = make_palette(s, v, oh)
        palettes_for_log.append((label, s, v, oh, palette))

        render = render_with_palette(palette)
        cropped = scb.crop_to_cube_row(render)
        target_w = 800
        scale = target_w / cropped.shape[1]
        target_h = int(round(cropped.shape[0] * scale))
        big = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        big_bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
        text = f"#{i+1}  {label}  s={s:.2f} v={v:.2f}" + (f" oh={int(oh)}" if oh else "")
        panels.append(scb.label_panel(big_bgr, text))

    cols = 2
    rows = [np.hstack(panels[r*cols:(r+1)*cols]) for r in range(len(panels)//cols)]
    full = np.vstack(rows)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweeps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "palette_variants.png")
    cv2.imwrite(out_path, full)
    print(f"\nSaved: {out_path}  ({full.shape[1]}w × {full.shape[0]}h)")

    # Dump the RGB values for each variant so the user can pick one and we
    # know what to paste into place.py.
    print("\n" + "=" * 70)
    for label, s, v, oh, palette in palettes_for_log:
        oh_str = f" oh={int(oh)}" if oh is not None else ""
        print(f"\n# {label}  (sat_lift={s:.2f}, val_lift={v:.2f}{oh_str})")
        for name, rgb in zip(NAMES, palette):
            r255 = int(round(rgb[0] * 255))
            g255 = int(round(rgb[1] * 255))
            b255 = int(round(rgb[2] * 255))
            print(f"#   {name:7} = ({r255:3}, {g255:3}, {b255:3})")


if __name__ == "__main__":
    main()

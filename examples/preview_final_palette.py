# Render the 6-cube row using the palette as currently baked into place.py.
# No monkey-patching of COLOR_PALETTE — this verifies the source edit took
# effect and produces the canonical reference shot.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import envs  # noqa: F401
import sweep_cube_brightness as scb  # noqa: E402

env = scb.make_env(n_distractors=5)
scb.lay_out_six_cubes(env, brightness_scale=1.0)
render = scb.capture(env)
env.close()

cropped = scb.crop_to_cube_row(render)
target_w = 1400
scale = target_w / cropped.shape[1]
target_h = int(round(cropped.shape[0] * scale))
big = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
big_bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
big_bgr = scb.label_panel(big_bgr, "FINAL palette (place.py, COLOR_PALETTE)  R B G Y P O")

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "sweeps", "final_palette.png")
cv2.imwrite(out, big_bgr)
print(f"Saved: {out}")

"""Measure the 6 cube hues under the real camera/lighting, for the colour-
selective distractor mask in infer_linux (keep table + goal cube, grey the rest).

Lay the 6 cubes on the table in a ROW, LEFT → RIGHT in colour-index order:

    red(0)  blue(1)  green(2)  yellow(3)  purple(4)  orange(5)

with NOTHING else coloured in the wrist-camera view. The tool finds the 6
saturated blobs, sorts them left-to-right, reads each one's median hue, and
writes hue_calib.json — which infer_linux / final_utils auto-load to override
the palette-derived hue defaults. Left-to-right assignment avoids the red/orange
hue ambiguity (their hues are very close).

    python -m final_utils.calib_colors                 # default /dev/video0
    python -m final_utils.calib_colors --camera_index 1
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer_linux as il
from infer_linux import Cv2Camera

COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]
CALIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hue_calib.json")


def _hue_dist(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera_index", type=int, default=il.CAMERA_INDEX)
    p.add_argument("--sat_min", type=int, default=il.DISTRACTOR_SAT_MIN,
                   help=f"min saturation for a pixel to count as a coloured cube (default {il.DISTRACTOR_SAT_MIN})")
    p.add_argument("--val_min", type=int, default=40, help="min value (drop dark/shadow pixels)")
    p.add_argument("--detect_w", type=int, default=320)
    p.add_argument("--frames", type=int, default=10, help="frames to average for stability")
    p.add_argument("--no_save", action="store_true")
    args = p.parse_args()

    cam = Cv2Camera(index=args.camera_index, width=1920, height=1080, fps=30)
    try:
        acc = None
        for _ in range(args.frames):
            f = np.asarray(cam.async_read(), dtype=np.float32)
            acc = f if acc is None else acc + f
        rgb = (acc / args.frames).astype(np.uint8)
    finally:
        cam.close()

    h = int(round(args.detect_w * rgb.shape[0] / rgb.shape[1]))
    rgb = cv2.resize(rgb, (args.detect_w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    colored = ((S >= args.sat_min) & (V >= args.val_min)).astype(np.uint8) * 255
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(colored, 8)
    blobs = sorted(range(1, num), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:6]
    if len(blobs) < 6:
        print(f"⚠ found only {len(blobs)} coloured blobs (need 6). Improve lighting / lower --sat_min "
              f"/ ensure all 6 cubes are in view and nothing else is coloured.")
        if not blobs:
            return

    measured = []
    for lab in blobs:
        m = labels == lab
        measured.append({"hue": float(np.median(H[m])), "sat": float(np.median(S[m])),
                         "val": float(np.median(V[m])), "area": int(stats[lab, cv2.CC_STAT_AREA]),
                         "cx": float(cents[lab][0]), "cy": float(cents[lab][1])})

    # Positional assignment: cubes are laid out left→right in colour-index order,
    # so the leftmost blob is red(0), next blue(1), ... rightmost orange(5).
    priors = il.GOAL_HUE_CV
    ok = len(measured) == 6
    hues = sorted(measured, key=lambda b: b["cx"]) if ok else [None] * 6

    print(f"\n{'color':8} {'prior':>5} {'measured':>9} {'sat':>5} {'val':>5} {'centroid':>14}")
    for c in range(6):
        b = hues[c]
        if b is None:
            print(f"{COLOR_NAMES[c]:8} {priors[c]:5d}   (missing)")
            continue
        flag = "  ⚠ far from prior" if _hue_dist(priors[c], b["hue"]) > 25 else ""
        print(f"{COLOR_NAMES[c]:8} {priors[c]:5d} {b['hue']:9.1f} {b['sat']:5.0f} {b['val']:5.0f} "
              f"({b['cx']:5.0f},{b['cy']:5.0f}){flag}")

    if ok and not args.no_save:
        out = {"hues": [hues[c]["hue"] for c in range(6)],
               "sat": [hues[c]["sat"] for c in range(6)],
               "names": COLOR_NAMES, "sat_min": args.sat_min}
        with open(CALIB_PATH, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n→ saved {CALIB_PATH}  (auto-loaded by infer_linux / final_utils)")
        print("  Verify the centroids match where each cube physically is; if red/orange are "
              "swapped, move them apart and re-run.")
    elif not ok:
        print("\nNot all colours assigned — not saving. Re-run with all 6 cubes clearly visible.")


if __name__ == "__main__":
    main()

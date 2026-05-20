"""Live HSV-based table masking for sim2real.

The sim renders a controlled background (no clutter behind the table); the real
camera sees whatever room/wall is behind the table edge. That mismatch can drag
the policy's domain-randomized features off-distribution. This script:

  1. seeds the table colour from a region of the frame you trust (click on the
     table, or use the bottom-strip fallback)
  2. builds an HSV mask around that seed, cleans it morphologically, and takes
     the largest connected component as the table region
  3. computes the table's average BGR colour from the table-coloured pixels
     only (the cube/gripper sit ON the table, NOT in the table-colour set,
     so they don't pollute the mean)
  4. paints every pixel OUTSIDE the table's convex hull with that average

Run from the SO101's initial pose so the wrist sees mostly table + the gripper.
The original "dominant-hue-across-whole-frame" mode was flaky because if the
background covers more pixels than the table, the wall becomes "dominant" and
we end up masking the table instead of the wall — exactly the inversion you
observed. Click-to-seed eliminates that.

Usage:
    python examples/table_mask_live.py
    python examples/table_mask_live.py --camera 1
    python examples/table_mask_live.py --table_hue 15 --hue_tol 12     # headless seed

Keys:
    LEFT CLICK on the "original" pane to seed the table colour from that pixel
    SPACE      freeze the current detection (next frames reuse the cached mask)
    R          release the freeze
    A          toggle bottom-strip auto-fallback ON / OFF
    Q / Esc    quit
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer_linux import Cv2Camera


@dataclass
class TableMask:
    hull_mask: np.ndarray        # uint8 0/255 — convex hull of table region
    table_mask: np.ndarray       # uint8 0/255 — pixels matching the table colour
    mean_bgr: np.ndarray         # shape (3,), avg of table pixels (BGR)
    seed_hsv: Tuple[int, int, int]   # median (H, S, V) of the seed window
    mode: str                    # "white" (low-sat) or "color" (hue-based)
    hull_polygon: np.ndarray     # contour, for debug overlay


def _seed_hsv(H, S, V, seed_xy, fallback_bottom_strip, win=11):
    """Median (H, S, V) of a window around the seed plus a representative
    seed POINT (x, y) on the table. Returns (hsv, point) or (None, None).
    The point is used to select the connected component the table belongs to."""
    h_img, w_img = H.shape
    if seed_xy is not None:
        x, y = int(seed_xy[0]), int(seed_xy[1])
        x0, x1 = max(0, x - win), min(w_img, x + win + 1)
        y0, y1 = max(0, y - win), min(h_img, y + win + 1)
        hh, ss, vv = H[y0:y1, x0:x1], S[y0:y1, x0:x1], V[y0:y1, x0:x1]
        point = (x, y)
    elif fallback_bottom_strip:
        y_lo = int(0.55 * h_img)
        x_lo, x_hi = int(0.30 * w_img), int(0.70 * w_img)
        hh, ss, vv = H[y_lo:, x_lo:x_hi], S[y_lo:, x_lo:x_hi], V[y_lo:, x_lo:x_hi]
        point = (w_img // 2, int(0.80 * h_img))   # bottom-center: very likely table
    else:
        return None, None
    if hh.size == 0:
        return None, None
    return (int(np.median(hh)), int(np.median(ss)), int(np.median(vv))), point


def detect_table(
    bgr: np.ndarray,
    seed_xy: Optional[Tuple[int, int]] = None,
    fallback_bottom_strip: bool = True,
    hue_tol: int = 14,
    sat_band: int = 45,
    val_band: int = 95,
    white_sat_thresh: int = 60,
    morph_kernel: int = 5,
    min_area_frac: float = 0.05,
) -> Optional[TableMask]:
    """Detect the table region adaptively from a seed colour (median HSV of a
    window around the click, or of the bottom-center strip):

      * WHITE/GRAY table (seed saturation < white_sat_thresh): hue is
        meaningless, so the table is the bright + desaturated region —
        S <= seed_S + sat_band AND V >= seed_V - val_band. (Real white tabletop.)
      * COLOURED table (seed saturation >= white_sat_thresh): hue-based —
        H within ±hue_tol of seed_H, with S/V floors.
    """
    if bgr is None or bgr.size == 0:
        return None
    h_img, w_img = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    seed, seed_pt = _seed_hsv(H, S, V, seed_xy, fallback_bottom_strip)
    if seed is None:
        return None
    sH, sS, sV = seed

    if sS < white_sat_thresh:
        mode = "white"
        s_max = min(255, sS + sat_band)
        v_min = max(0, sV - val_band)
        table_pixels = (S <= s_max) & (V >= v_min)
    else:
        mode = "color"
        lo, hi = sH - hue_tol, sH + hue_tol
        if lo < 0:
            h_mask = (H >= (180 + lo)) | (H <= hi)
        elif hi > 179:
            h_mask = (H >= lo) | (H <= (hi - 180))
        else:
            h_mask = (H >= lo) & (H <= hi)
        s_min = max(0, sS - sat_band)
        v_min = max(0, sV - val_band)
        table_pixels = h_mask & (S >= s_min) & (V >= v_min)

    # Morphological cleanup: close holes (cubes/gripper on the table), then
    # open to drop speckle.
    kern = np.ones((morph_kernel, morph_kernel), np.uint8)
    table_u8 = table_pixels.astype(np.uint8) * 255
    table_u8 = cv2.morphologyEx(table_u8, cv2.MORPH_CLOSE, kern, iterations=3)
    table_u8 = cv2.morphologyEx(table_u8, cv2.MORPH_OPEN,  kern, iterations=1)

    # Pick the connected component CONTAINING the seed point (the table you
    # clicked / the bottom-center). Falls back to the largest component if the
    # seed lands on background. This avoids grabbing a bright curtain/wall blob
    # that happens to be bigger than the table.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(table_u8, 8)
    if num <= 1:
        return None
    sx, sy = int(np.clip(seed_pt[0], 0, w_img - 1)), int(np.clip(seed_pt[1], 0, h_img - 1))
    seed_label = int(labels[sy, sx])
    if seed_label == 0:
        # Seed not on any component — fall back to largest.
        seed_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[seed_label, cv2.CC_STAT_AREA] < min_area_frac * H.size:
        return None
    table_mask = (labels == seed_label).astype(np.uint8) * 255

    # Convex hull of the table → the polygon "above which everything is on/over the table"
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(np.concatenate(contours))
    hull_mask = np.zeros_like(table_mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    # Mean BGR from table-coloured pixels ONLY (NOT the hull — cube + gripper
    # live inside the hull but are not table-coloured, so they'd skew the mean).
    mean_bgr = bgr[table_mask > 0].reshape(-1, 3).mean(axis=0).astype(np.float32)

    return TableMask(
        hull_mask=hull_mask, table_mask=table_mask, mean_bgr=mean_bgr,
        seed_hsv=(sH, sS, sV), mode=mode, hull_polygon=hull,
    )


def paint_outside_hull(bgr: np.ndarray, m: TableMask) -> np.ndarray:
    """Mode A: pixels OUTSIDE the convex hull → mean. Inside (table surface +
    cube + gripper) is preserved. This is "scene becomes infinite table"."""
    out = bgr.copy()
    out[m.hull_mask == 0] = m.mean_bgr
    return out


def paint_inside_hull(bgr: np.ndarray, m: TableMask) -> np.ndarray:
    """Mode B: pixels INSIDE the convex hull → mean. Outside is preserved.
    This flattens the table to a uniform colour AND hides cube + gripper."""
    out = bgr.copy()
    out[m.hull_mask > 0] = m.mean_bgr
    return out


def paint_outside_table(bgr: np.ndarray, m: TableMask) -> np.ndarray:
    """Mode C: pixels NOT in the table HSV range → mean. This keeps ONLY
    visibly table-coloured pixels and flattens everything else (including
    cube + gripper, which sit on top of the table but are not table-coloured)."""
    out = bgr.copy()
    out[m.table_mask == 0] = m.mean_bgr
    return out


def overlay_debug(bgr: np.ndarray, m: TableMask, click_xy: Optional[Tuple[int, int]] = None) -> np.ndarray:
    out = bgr.copy()
    cv2.drawContours(out, [m.hull_polygon], -1, (0, 255, 0), 2)
    # Seed-colour swatch (top-right): the actual median HSV of the seed window.
    sH, sS, sV = m.seed_hsv
    swatch_hsv = np.full((60, 60, 3), 0, dtype=np.uint8)
    swatch_hsv[..., 0] = sH
    swatch_hsv[..., 1] = sS
    swatch_hsv[..., 2] = sV
    swatch_bgr = cv2.cvtColor(swatch_hsv, cv2.COLOR_HSV2BGR)
    out[10:70, -70:-10] = swatch_bgr
    cv2.putText(out, f"{m.mode} HSV {sH},{sS},{sV}", (out.shape[1] - 200, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    if click_xy is not None:
        cv2.drawMarker(out, click_xy, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    return out


def tile(*imgs: np.ndarray) -> np.ndarray:
    return np.concatenate(imgs, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--display_width", type=int, default=420,
                   help="per-pane width; 4 panes shown side-by-side")
    p.add_argument("--hue_tol", type=int, default=14,
                   help="(colour-table mode) ±hue degrees accepted around seed")
    p.add_argument("--sat_band", type=int, default=45,
                   help="saturation band around the seed (both modes)")
    p.add_argument("--val_band", type=int, default=95,
                   help="how far below the seed brightness still counts as table")
    p.add_argument("--white_sat_thresh", type=int, default=60,
                   help="seed saturation below this → white/gray table mode")
    args = p.parse_args()

    cam = Cv2Camera(index=args.camera, width=args.width, height=args.height, fps=args.fps)
    print(f"Opened /dev/video{args.camera} at {args.width}×{args.height}@{args.fps}")
    print("Keys: LEFT-CLICK on a table pixel to seed, SPACE freeze, R release, "
          "A toggle auto-strip, Q/Esc quit")

    aspect = args.height / args.width
    pane_w = args.display_width
    pane_h = int(round(pane_w * aspect))

    seed_xy: Optional[Tuple[int, int]] = None
    auto_strip = True
    frozen: Optional[TableMask] = None

    def on_mouse(event, x, y, flags, param):
        nonlocal seed_xy, frozen
        if event == cv2.EVENT_LBUTTONDOWN:
            # 4 panes side-by-side, each pane_w wide; map x back to the
            # original-frame coord so a click on any pane works.
            sx = x % pane_w
            seed_xy = (sx, y)
            frozen = None  # re-detect immediately with new seed
            print(f"seeded from click  (pane click @ ({x}, {y}) → frame ({sx}, {y}))")

    win = "table mask: orig | table_mask | Mode A bg->mean | Mode B table->mean"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    # Realize the window with a dummy frame BEFORE setMouseCallback — some
    # OpenCV Qt builds return a NULL window handle until the first imshow.
    cv2.imshow(win, np.zeros((pane_h, pane_w * 4, 3), dtype=np.uint8))
    cv2.waitKey(1)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        rgb = cam.async_read()
        bgr_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr_full, (pane_w, pane_h), interpolation=cv2.INTER_AREA)

        if frozen is not None:
            m = frozen
        else:
            m = detect_table(
                bgr,
                seed_xy=seed_xy,
                fallback_bottom_strip=auto_strip and seed_xy is None,
                hue_tol=args.hue_tol, sat_band=args.sat_band,
                val_band=args.val_band, white_sat_thresh=args.white_sat_thresh,
            )

        if m is not None:
            # Pane 1: original with hull outline + click marker + hue swatch
            orig_with_overlay = overlay_debug(bgr, m, click_xy=seed_xy)
            # Pane 2: green overlay marking the table_mask pixels (visual check
            # of which pixels the algorithm thinks are table-coloured)
            table_overlay = bgr.copy()
            green = np.zeros_like(bgr)
            green[..., 1] = 255
            mix = (m.table_mask > 0)[..., None]
            table_overlay = np.where(mix, (bgr * 0.4 + green * 0.6).astype(np.uint8), bgr)
            # Pane 3: Mode A — paint OUTSIDE hull with mean (background → table)
            mode_a = paint_outside_hull(bgr, m)
            # Pane 4: Mode B — paint INSIDE hull with mean (table area → flat colour)
            mode_b = paint_inside_hull(bgr, m)
            display = tile(orig_with_overlay, table_overlay, mode_a, mode_b)
            tag = "FROZEN" if frozen is not None else "LIVE"
            mean = m.mean_bgr
            seed = "click" if seed_xy is not None else ("strip" if auto_strip else "cli")
            sH, sS, sV = m.seed_hsv
            txt = (f"{tag} seed={seed} mode={m.mode} HSV=({sH},{sS},{sV}) "
                   f"meanBGR=({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f})  "
                   f"panes: orig | table_mask | A bg->mean | B table->mean")
            cv2.putText(display, txt, (8, display.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            blanks = [np.zeros_like(bgr) for _ in range(3)]
            display = tile(bgr, *blanks)
            txt = "no table detected — left-click a table pixel to seed"
            cv2.putText(display, txt, (8, display.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" ") and m is not None:
            frozen = m
            print(f"FROZEN  mode={m.mode}  seed HSV={m.seed_hsv}  mean BGR={tuple(m.mean_bgr.astype(int))}")
        if key == ord("r"):
            frozen = None
            print("released — live detection")
        if key == ord("a"):
            auto_strip = not auto_strip
            print(f"auto bottom-strip fallback: {auto_strip}")

    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

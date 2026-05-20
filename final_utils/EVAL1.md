# Eval 1 — Pick & Place

Pick up the cube of a queried colour and drop it into the bowl. The vision RL
policy aligns to the cube; a hardcoded FK/IK layer does the precise grasp, carry,
and release. Success = the cube is grasped, carried to the bowl, and released.

All commands run from the repo root with the env active:

```bash
conda activate squint
cd /home/team44/squint
```

Hardware: SO101 follower on `/dev/ttyACM0` + wrist camera on `/dev/video0`
(pass `--robot_port` / `--camera_index` if different).

---

## One-time calibration (per rig / per setup)

These write small JSON files at the repo root that the eval **auto-loads**. Redo
them if you move the camera, change the lighting, or change the table.

### 1. Table height vs. reach — `table_z_calib.json`

The real arm's geometry drifts from the model with extension, so the FK height
that means "touching the table" changes with reach. Calibrate it:

```bash
python examples/table_z_calib.py
```

The arm goes limp (gripper held closed). **Slide the closed fingertip across the
table from near the base out to full reach** (and back), then `Ctrl+C`. It fits
`z_table(r)` and saves it. Without this file the eval assumes a flat table and
far cubes won't be grasped well.

### 2. Cube colours — `hue_calib.json`

The policy is told a colour; the camera mask keeps the table + that colour's cube
and greys everything else (bowl, other cubes). Measure the real cube hues:

> **Lay the 6 cubes in a row, LEFT → RIGHT, in this exact order:**
>
> ```
> red(0)   blue(1)   green(2)   yellow(3)   purple(4)   orange(5)
> ```
>
> Nothing else coloured in the camera view (no bowl). Spread them out a little.

```bash
python -m final_utils.calib_colors
```

It reads the 6 cubes left-to-right and saves their hues. **Check the printed
table** — the `centroid` x should increase down the rows (left→right) and no row
should be flagged `⚠ far from prior`. If something's off, fix the order/spacing
and re-run. Left-to-right ordering is what disambiguates red vs. orange (their
hues nearly overlap).

### 3. Bowl position — taught live (per bowl placement)

The place step flies to an absolute `(x, y)` in the **robot base frame** (FK
frame), which is *not* a frame you can eyeball. Teach it:

```bash
python -m final_utils.teach_bowl_xy
```

The arm goes limp (gripper closed). **Move the gripper tip directly over the bowl
centre**, then `Ctrl+C`. It prints the base-frame `x y` to pass to the run.

---

## Run the eval

```bash
python -m final_utils.pick_place --goal_color 0 --bowl_xy <x> <y>
```

- `--goal_color`: `0 red · 1 blue · 2 green · 3 yellow · 4 purple · 5 orange`
- `--bowl_xy`: the `x y` from step 3 (metres, robot base frame).
- Defaults already set: `--action_scale 0.45`, `--episode_steps 1000`, bundled
  checkpoint `final_utils/pick_place_policy.pt`.

Exit code is **0 on success** (cube placed) and **1 if it didn't finish**, so it
scripts cleanly:

```bash
python -m final_utils.pick_place --goal_color 0 --bowl_xy 0.18 -0.06 && echo "EVAL1 PASS"
```

From Python:

```python
from final_utils import pick_and_place
ok = pick_and_place(goal_color=0, bowl_xy=(0.18, -0.06))
```

---

## What happens in a run

1. **Approach** — policy drives the arm and gripper toward the cube.
2. **Gate** — fires when the descent stalls near the table (or hits the height
   gate). The Rerun viewer shows the masked policy input (`camera/policy_input`)
   so you can confirm the bowl/other cubes are greyed and the goal cube is kept.
3. **Grasp** — nudge to centre, full close, verify by the gripper stall angle,
   hold, lift 5 cm. A miss retreats and retries.
4. **Place** — IK to the bowl `(x, y)` at 10 cm above the table, hold 0.5 s, open
   to drop. → success.

## Useful flags

| Flag | Default | What |
|------|---------|------|
| `--place_z` | `0.10` | drop height above the table (m) |
| `--place_open_wait_s` | `0.5` | hold over bowl before opening |
| `--place_speed` | `0.30` | carry speed to the bowl (m/s) |
| `--no-viz` | (on) | disable the Rerun viewer |
| `--robot_port` / `--camera_index` | `/dev/ttyACM0` / `0` | hardware overrides |

## Troubleshooting

- **Doesn't grasp / closes empty** — check the live `above_table` print and the
  masked viewer; re-run table-z calibration if the height looks wrong.
- **Drops in the wrong spot** — `bowl_xy` frame is wrong; re-teach with
  `teach_bowl_xy` (don't eyeball the numbers).
- **Bowl/other cube still distracts** — re-run `calib_colors`; tune
  `DISTRACTOR_SAT_MIN` (what counts as coloured) or `GOAL_HUE_TOL` (hue kept
  around the goal) in `infer_linux.py`.

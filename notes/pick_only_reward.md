# Pick-only reward mode for `Lift` env

All changes in [`envs/lift.py`](../envs/lift.py). Adds a `pick_only_reward` bool that
reshapes the reward function and success criterion so the policy is trained to
**pick the cube only** (no return-to-rest-pose phase), terminating early on a
1-second sustained "grasped + nearly stationary" condition.

## Summary of changes

### 1. Constructor (new kwargs)
```python
pick_only_reward: bool = False
pick_stable_speed_threshold: float = 0.01   # m/s (1 cm/s)
pick_stable_duration_s: float = 1.0         # seconds
```

Default `False` → fully backward-compatible with existing training runs.

### 2. `place_reward` removed unconditionally
The `exp(-2 · dist_to_rest_qpos) · is_item_grasped` term is gone from
`compute_dense_reward` in **both** modes. The original "lift task" rewarded
returning to the SO101 `start` keyframe pose after grasp — we no longer want
that signal.

### 3. Pre-existing bug fix (CombinedController access)
`self.agent.controller._target_qpos` does not exist — the SO101 wraps its arm
controller in a dict, producing a `CombinedController`. Fixed three call sites
to use the same pattern as [`envs/place.py:1230`](../envs/place.py#L1230):

```python
self.agent.controller.controllers["arm"]._target_qpos
```

(The env couldn't reset before this fix.)

### 4. Per-env stability counter
`self._grasp_slow_counter`: int32 tensor of shape `(num_envs,)`. Initialized in
`_load_scene`, reset to 0 per-env in `_initialize_episode`. Only consulted when
`pick_only_reward=True`.

### 5. Success criterion (pick-only mode)
```python
item_speed = ||cube.linear_velocity||
is_item_slow = item_speed < pick_stable_speed_threshold     # < 1 cm/s
stable_grasp = is_grasping(cube) & is_item_slow
counter = where(stable_grasp, counter + 1, 0)               # strict consecutive
success = counter >= round(pick_stable_duration_s * control_freq)
```

At `control_freq=10 Hz` and `duration=1.0 s` → 10 consecutive control steps.
ManiSkill auto-terminates the episode on `info["success"]=True`
([`sapien_env.py:1054-1058`](file:///home/team44/.conda/envs/squint/lib/python3.10/site-packages/mani_skill/envs/sapien_env.py#L1054-L1058)) — no extra plumbing needed.

Default mode keeps the original `item_lifted & is_item_grasped & reached_rest_qpos`.

### 6. Reward (pick-only mode)
```python
reach        = 1 - tanh(5 · ||tcp - cube||)                 # k=5
is_grasped   = info["is_item_grasped"].float()              # binary (0.5N + angle)
# Force-graded gate: smooth ramp 0 → 1 as weaker finger reaches 5 N on cube.
min_force    = min(||l_contact_force||, ||r_contact_force||)
force_gate   = (min_force / 5.0).clamp(0, 1)
gripper_term = force_gate · exp(-5 · |gripper_target_qpos|)  # target → 0 = closed hard

reward  = (1 - is_grasped) · reach           # reach drops out after grasp
        + is_grasped                          # grasp signal (binary)
        + gripper_term                        # graded squeeze × close-target
        - 3.0 · robot_touching_table          # arm-dragging penalty
        + success · (max_steps - elapsed) · 2.0   # terminal bonus

return reward / 3                              # via compute_normalized_dense_reward
```

The terminal bonus uses `per_step_peak = 2.0` (grasp + close = 2 post-grasp) ×
remaining steps, so terminating early matches a hypothetical "continue at peak"
trajectory: both yield total un-normalized return ≈ `2 · max_steps`.

`max_steps` is read from `mani_skill.utils.registration.REGISTERED_ENVS[env_id].max_episode_steps`
since gymnasium's `env.spec.max_episode_steps` is `None` for ManiSkill envs.

### 7. Default mode reward changes
- `place_reward` term removed (see §2).
- All other terms unchanged: `reach + is_grasped - 3·table - 1·(~item_lifted)`.

## Tuned parameter values (chosen via iterative selection)

| Knob | Value | Notes |
|---|---|---|
| reach sharpness `k` in `1 - tanh(k·d)` | **5** | reward 0.76 @ 5 cm, 0.46 @ 10 cm |
| close sharpness `k` in `exp(-k·|target|)` | **5** | reward 0.61 @ 0.1 rad, 0.37 @ 0.2 rad |
| force-gate saturation | **5 N** | min(l,r) force at which gate hits 1.0 |
| reach reward post-grasp | **dropped** | `(1 - is_grasped) · reach` |
| close reward pre-grasp | **only when fingers in contact** | gated by `force_gate`, not by `is_grasped` |
| close reward gate type | **force-graded** | smooth ramp via `min_finger_force / 5N` |
| robot-table contact penalty | **−3.0** | matches original default-mode magnitude |
| arm velocity / action smoothness penalty | **none** | handled at controller level |
| first-grasp sparse bonus | **none** | dense reward only |
| stability detection | **strict consecutive** | any bad step → counter reset to 0 |
| stability speed threshold | **1 cm/s** (0.01 m/s) | |
| stability duration | **1.0 s** (10 steps @ 10 Hz) | |
| failure / early-termination | **none** | only success or timeout |
| lift-gating on success | **none** | grasped + slow only (no z-height check) |
| final reward | **remaining × per-step peak** | `(max_steps - elapsed) · 2.0` |
| reward weights (reach / grasp / close) | **1 / 1 / 1** | |
| `compute_normalized_dense_reward` divisor | **3** | unchanged (under-normalizes new peak of 2, but consistent) |

## How to enable

```python
import gymnasium as gym
import envs

env = gym.make(
    "SO101LiftCube-v1",
    num_envs=64,
    pick_only_reward=True,
    # optional overrides:
    # pick_stable_speed_threshold=0.01,
    # pick_stable_duration_s=1.0,
)
```

`SO101LiftCan-v1` also inherits the same behavior (both subclass `Lift`).

## Smoke tests run

- Default mode reset + 3 random steps → no error, reward in expected range.
- Pick-only mode reset + 3 random steps → no error, counter stays at 0 (cube not
  grasped under random actions).
- Pick-only forced-success path: counter manually set to 1, `is_grasping`
  patched to return True for env 0 → next step:
  - `counter[0] = 2 ≥ stable_steps_required` ✓
  - `success[0] = True`, `terminated[0] = True` ✓
  - `reward[0] ≈ 99` (normalized) = `(1 + 149 · 2) / 3` ✓
- `SO101LiftCan-v1` with `pick_only_reward=True` resets and steps cleanly.

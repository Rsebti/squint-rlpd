#!/usr/bin/env python3
"""
Collect 50 RLPD demos via scripted FK/IK (Tommaso's V3 recipe, ported from
fedecomi04/squint:tom-separating-cubes commit c6f9808).

num_envs=8 batched for speed. Output matches the v2 HDF5 schema and is
consumed by rlpd_utils._load_h5_v2 at training time.

Sanity videos: 1 per color (6 total), saved alongside the h5.

Usage:
    python scripts/collect_rlpd_demos.py
    # writes /tmp/rlpd_50demos/{demos.h5, meta.json, sanity_*.mp4}

Notable deviations from the spec (all flagged in the h5 attrs):
  - control_mode='pd_joint_pos' (absolute) instead of pd_joint_target_delta_pos.
    Convert at load time: delta[t] = action[t] - action[t-1] with
    action[-1]=QPOS_START, normalize by [0.05]*5+[0.2]. Done by
    rlpd_utils._load_h5_v2.
  - No ManiSkillVectorEnv wrapper during collection (A/B'd: MSVecEnv kills
    grasp dynamics under scripted IK).
  - domain_randomization=False (DR drops scripted-IK grasp rate <5%). Trainer
    re-applies visual jitter via ColorJitterWrapper.
"""
import sys, os, math, json, time, datetime, subprocess
import numpy as np, torch, gymnasium as gym, h5py, cv2

# Resolve repo root from the script's own location so this works regardless
# of where the repo is cloned (vs Tommaso's hardcoded /home/shadeform/squint).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
import envs  # noqa: F401 — registers SO101PlaceCube-v1
from so101_fk import tcp_pos, nudge_arm_joints, finger_positions

OUT_DIR = '/tmp/rlpd_50demos'
OUT_H5  = f'{OUT_DIR}/demos.h5'
OUT_META= f'{OUT_DIR}/meta.json'
NUM_DEMOS_TARGET = 50
NUM_COLORS  = 6
BATCH_SIZE  = 8
MAX_BATCHES = 35           # ~30% rate → ~21 batches for 50; 35 = comfortable margin
PER_COLOR_MAX = (NUM_DEMOS_TARGET + NUM_COLORS - 1) // NUM_COLORS  # 9
IMG_H, IMG_W = 80, 144
N_ARM = 5

GRIPPER_OPEN_FULL = np.float32(120 * math.pi / 180)
GRIPPER_OPEN_DESC = np.float32(60  * math.pi / 180)
GRIPPER_CLOSED    = np.float32(5   * math.pi / 180)
GRIPPER_CLOSED_F  = float(GRIPPER_CLOSED)

QPOS_START = np.array([
    0.0, -80.791*math.pi/180, 36.747*math.pi/180,
    86.901*math.pi/180, -82.154*math.pi/180, GRIPPER_OPEN_FULL,
], dtype=np.float64)

T_HOME, T_MOVE, T_DESC = 30, 60, 65
T_GRIP, T_GRASP_HOLD   = 100, 100
T_LIFT, T_TRANS, T_HOVER, T_REL = 110, 120, 35, 50

COLOR_NAMES = ['red','blue','green','yellow','purple','orange']

# ─── IK + face-to-face wrist alignment ───────────────────────────────────────
def solve_ik(q0, target_tcp, n_outer=200, step_frac=0.05, tol=0.002):
    q = np.array(q0, dtype=np.float64).copy()
    for _ in range(n_outer):
        cur = tcp_pos(q); e = target_tcp - cur; err = np.linalg.norm(e)
        if err < tol: return q
        step = e * min(1.0, step_frac/err)
        q = q + nudge_arm_joints(q, step, max_joint_step=0.30, iters=8)
    return q

def closed_jaw_xy_angle(q):
    qc = q.copy(); qc[5] = GRIPPER_CLOSED_F
    f1, f2 = finger_positions(qc)
    v = (f2 - f1)[:2]
    if np.linalg.norm(v) < 1e-5: return 0.0
    return math.atan2(v[1], v[0])

def cube_theta_from_quat(q_wxyz):
    return 2 * math.atan2(q_wxyz[3], q_wxyz[0])

def best_face_delta(cur, theta):
    best, ba = 0.0, math.inf
    for k in range(4):
        d = (theta + k*math.pi/2) - cur
        d = ((d + math.pi) % (2*math.pi)) - math.pi
        if abs(d) < ba: best, ba = d, abs(d)
    return best

def align_wrist_roll(q, theta, lo=-2.74, hi=2.84):
    delta = best_face_delta(closed_jaw_xy_angle(q), theta)
    qn = q.copy()
    qn[4] = max(lo+0.05, min(hi-0.05, qn[4] + delta))
    return qn

def plan_env(cube_pos, cube_quat, bowl_pos, cube_half, bowl_hz, q_init):
    theta = cube_theta_from_quat(cube_quat)
    cube_top = cube_pos[2] + cube_half
    bowl_rim = bowl_pos[2] + 2 * bowl_hz
    tcp_pre   = np.array([cube_pos[0], cube_pos[1], cube_top + 0.10])
    tcp_grasp = np.array([cube_pos[0], cube_pos[1], cube_pos[2] - 0.003])
    tcp_bowl  = np.array([bowl_pos[0], bowl_pos[1], bowl_rim + 0.04])
    q_pre = solve_ik(q_init, tcp_pre)
    q0g = q_pre.copy(); q0g[5] = GRIPPER_CLOSED
    q_grasp = solve_ik(q0g, tcp_grasp)
    q_grasp = align_wrist_roll(q_grasp, theta)
    q_grasp = solve_ik(q_grasp, tcp_grasp)
    q_grasp = align_wrist_roll(q_grasp, theta)
    q_lift = q_grasp.copy(); q_lift[1] -= 0.55; q_lift[2] += 0.20
    q_bowl = solve_ik(q_lift, tcp_bowl)
    return dict(q_pre=q_pre, q_grasp=q_grasp, q_lift=q_lift, q_bowl=q_bowl)

def cos_t(i, n): return (1 - math.cos(math.pi * i / max(n-1, 1))) / 2

def lerp_seq(q0, q1, n, hold_n):
    out = [((1-cos_t(i,n))*q0 + cos_t(i,n)*q1) for i in range(n)]
    out += [q1.copy() for _ in range(hold_n)]
    return out

def build_abs_sequence(plan, q_init):
    qi = q_init.astype(np.float32);    qi[5] = GRIPPER_OPEN_FULL
    qpre  = plan['q_pre'].astype(np.float32);   qpre[5]  = GRIPPER_OPEN_FULL
    qg_op = plan['q_grasp'].astype(np.float32); qg_op[5] = GRIPPER_OPEN_DESC
    qg_cl = plan['q_grasp'].astype(np.float32); qg_cl[5] = GRIPPER_CLOSED
    ql    = plan['q_lift'].astype(np.float32);  ql[5]    = GRIPPER_CLOSED
    qb    = plan['q_bowl'].astype(np.float32);  qb[5]    = GRIPPER_CLOSED
    qr    = plan['q_bowl'].astype(np.float32);  qr[5]    = GRIPPER_OPEN_FULL
    seq  = lerp_seq(qi,     qpre,  T_MOVE + T_HOME, T_HOME)
    seq += lerp_seq(qpre,   qg_op, T_DESC, 25)
    seq += lerp_seq(qg_op,  qg_cl, T_GRIP, T_GRASP_HOLD)
    seq += lerp_seq(qg_cl,  ql,    T_LIFT, T_HOME)
    seq += lerp_seq(ql,     qb,    T_TRANS, T_HOVER)
    seq += lerp_seq(qb,     qr,    T_REL,  T_HOME)
    return np.stack(seq, axis=0).astype(np.float32)

# ─── Env: gym.make only (no ManiSkillVectorEnv) ─────────────────────────────
print(f"Building env (V3 recipe, num_envs={BATCH_SIZE}, no MSVecEnv)...")
env = gym.make(
    'SO101PlaceCube-v1',
    num_envs=BATCH_SIZE,
    obs_mode='rgb',
    render_mode='all',
    sim_backend='gpu',
    sensor_configs=dict(width=640, height=360),
    domain_randomization=False,
    n_distractors=0,
    use_real_bowl=True,
    control_mode='pd_joint_pos',
    sim_freq=100, control_freq=10,
    pick_only_reward=False, split_only_reward=False, action_smooth_coef=0.0,
)
ue = env.unwrapped
dev = torch.device('cuda:0')

obs0, _ = env.reset(seed=0)

def make_state(obs_d):
    """Concat to alphabetical-key order to mirror FlattenRGBDObservationWrapper.
       Returns (B, 15) float32: bowl_xyz(3) + goal_color(6) + noisy_qpos(6)."""
    parts = []
    a = obs_d['agent']
    for k in sorted(a.keys()):
        v = a[k]
        if torch.is_tensor(v):
            v = v.reshape(v.shape[0], -1) if v.ndim > 1 else v.unsqueeze(-1)
            parts.append(v)
    return torch.cat(parts, dim=-1).cpu().numpy().astype(np.float32)

def make_rgb_downsampled(obs_d):
    """Take obs sensor rgb (B,360,640,3) → (B, IMG_H, IMG_W, 3) uint8."""
    rgb_raw = obs_d['sensor_data']['base_camera']['rgb']
    if torch.is_tensor(rgb_raw): rgb_raw = rgb_raw.cpu().numpy()
    rgb_raw = np.asarray(rgb_raw)
    B = rgb_raw.shape[0]
    out = np.zeros((B, IMG_H, IMG_W, 3), dtype=np.uint8)
    for i in range(B):
        out[i] = cv2.resize(rgb_raw[i], (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return out

state15_dim = make_state(obs0).shape[-1]
print(f"  raw state_dim = {state15_dim}  (will augment to 21 with target_qpos at save time)")

# ─── Main loop ──────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
demos = []
color_counts = [0]*NUM_COLORS
saved_video_for_color = set()

def pick_colors():
    sc = sorted(range(NUM_COLORS), key=lambda c: color_counts[c])
    return [sc[i % NUM_COLORS] for i in range(BATCH_SIZE)]

t0 = time.time()
for batch_i in range(MAX_BATCHES):
    if len(demos) >= NUM_DEMOS_TARGET: break
    seed = batch_i * 1000 + 7
    obs, _ = env.reset(seed=seed)
    qstart_act = QPOS_START[np.newaxis].astype(np.float32).repeat(BATCH_SIZE, axis=0)

    target_colors = pick_colors()
    color_tensor = torch.tensor(target_colors, device=dev, dtype=torch.long)
    ue.goal_color_idx[:] = color_tensor
    ue._set_actor_palette_color(ue.item,
                                 torch.arange(BATCH_SIZE, device=dev),
                                 color_tensor)

    for _ in range(60):
        obs, _, _, _, _ = env.step(qstart_act)

    ue.goal_color_idx[:] = color_tensor
    ue._set_actor_palette_color(ue.item,
                                 torch.arange(BATCH_SIZE, device=dev),
                                 color_tensor)
    obs, _, _, _, _ = env.step(qstart_act)

    cube_p = ue.item.pose.p.cpu().numpy()
    cube_q = ue.item.pose.q.cpu().numpy()
    bowl_p = ue.bin.pose.p.cpu().numpy()
    cube_half = ue.item_half_sizes[0].cpu().item()
    bowl_hz = getattr(ue, 'bowl_half_z', 0.0265)
    q_init = ue.agent.robot.get_qpos()[:, :6].cpu().numpy()

    abs_seqs = []
    for i in range(BATCH_SIZE):
        plan = plan_env(cube_p[i], cube_q[i], bowl_p[i], cube_half, bowl_hz,
                         q_init[i].astype(np.float64))
        abs_seqs.append(build_abs_sequence(plan, q_init[i]))
    actions_batch = np.stack(abs_seqs, axis=1)
    T = actions_batch.shape[0]

    record_hires = any(c not in saved_video_for_color for c in target_colors)

    rgb_buf, st_buf, act_buf, rew_buf, done_buf = [], [], [], [], []
    hires_buf = []
    # Diagnostics: track per-env max cube z (was it ever lifted?) and whether
    # the cube was grasped at any point during the episode.
    max_cube_z = cube_p[:, 2].copy()
    ever_grasped = np.zeros(BATCH_SIZE, dtype=bool)
    for t in range(T):
        rgb_buf.append(make_rgb_downsampled(obs))
        st_buf.append(make_state(obs))
        if record_hires:
            r = ue.render()
            if torch.is_tensor(r): r = r.cpu().numpy()
            hires_buf.append(np.asarray(r))
        a = actions_batch[t]
        act_buf.append(a.copy())
        obs, rew, term, trunc, info = env.step(a)
        rew_buf.append(rew.cpu().numpy() if torch.is_tensor(rew) else np.asarray(rew))
        done_buf.append((term | trunc).cpu().numpy() if torch.is_tensor(term)
                          else np.asarray(term | trunc))
        # Track lift + grasp for diagnostics
        _cz = ue.item.pose.p[:, 2].detach().cpu().numpy()
        max_cube_z = np.maximum(max_cube_z, _cz)
        try:
            _g = ue.agent.is_grasping(ue.item).detach().cpu().numpy().astype(bool)
            ever_grasped = ever_grasped | _g
        except Exception:
            pass

    rgb_arr   = np.stack(rgb_buf,  axis=1)
    st15_arr  = np.stack(st_buf,   axis=1)
    act_arr   = np.stack(act_buf,  axis=1)
    rew_arr   = np.stack(rew_buf,  axis=1)
    done_arr  = np.stack(done_buf, axis=1)

    final_cube_xy = ue.item.pose.p[:, :2].cpu().numpy()
    final_cube_z  = ue.item.pose.p[:, 2].cpu().numpy()
    dist = np.linalg.norm(final_cube_xy - bowl_p[:, :2], axis=1)

    # ── Diagnostic (first 2 batches): pinpoint where the pipeline breaks ──
    if batch_i < 2:
        cube_half_dbg = ue.item_half_sizes[0].cpu().item()
        lift_thresh = cube_p[:, 2] + 0.03  # 3cm above spawn = "was lifted"
        print(f"  [diag b{batch_i}] cube_half={cube_half_dbg:.4f} "
              f"bowl_z={bowl_p[0,2]:.3f} bowl_hz={getattr(ue,'bowl_half_z',0.0265):.4f}")
        for i in range(BATCH_SIZE):
            print(f"    env{i}: spawn_z={cube_p[i,2]:.3f} max_z={max_cube_z[i]:.3f} "
                  f"final_z={final_cube_z[i]:.3f} dist_bowl={dist[i]:.3f} "
                  f"grasped={bool(ever_grasped[i])} "
                  f"lifted={bool(max_cube_z[i] > lift_thresh[i])}")

    n_added = 0
    for i in range(BATCH_SIZE):
        in_bowl = (dist[i] < 0.05) and (final_cube_z[i] > bowl_p[i, 2])
        if not in_bowl: continue
        color = target_colors[i]
        if color_counts[color] >= PER_COLOR_MAX: continue

        # Augment state 15→21: insert target_qpos (= previous action) at index 3.
        tq = np.zeros((T, 6), dtype=np.float32)
        tq[0]  = q_init[i].astype(np.float32)
        tq[1:] = act_arr[i, :-1]
        state21 = np.concatenate([st15_arr[i, :, :3], tq, st15_arr[i, :, 3:]], axis=1)

        demo_idx = len(demos)
        demos.append(dict(
            rgb=rgb_arr[i], state=state21, actions=act_arr[i],
            rewards=rew_arr[i], terminals=done_arr[i],
            color_idx=int(color), cube_pos=cube_p[i].tolist(),
            bowl_pos=bowl_p[i].tolist(), seed=int(seed),
            return_sum=float(rew_arr[i].sum()),
        ))
        color_counts[color] += 1
        n_added += 1

        if record_hires and color not in saved_video_for_color:
            cname = COLOR_NAMES[color]
            out_mp4 = os.path.join(OUT_DIR, f'sanity_{cname}_demo{demo_idx:03d}.mp4')
            H, W = hires_buf[0].shape[1], hires_buf[0].shape[2]
            proc = subprocess.Popen(
                ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
                 '-s', f'{W}x{H}', '-r', '30', '-i', '-',
                 '-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart', out_mp4],
                stdin=subprocess.PIPE)
            for bf in hires_buf:
                proc.stdin.write(cv2.cvtColor(bf[i], cv2.COLOR_RGB2BGR).tobytes())
            proc.stdin.close(); proc.wait()
            saved_video_for_color.add(color)
            print(f"    sanity video → {out_mp4}")

        if len(demos) >= NUM_DEMOS_TARGET: break

    print(f"  batch {batch_i:2d}  seed={seed:5d}  +{n_added}  total {len(demos):2d}/{NUM_DEMOS_TARGET}  "
          f"per_color={color_counts}  ({time.time()-t0:.0f}s)")

env.close()
print(f"\nFinal: {len(demos)} demos in {time.time()-t0:.0f}s")
print(f"Per-color: {color_counts}")
print(f"Sanity videos saved for colors: {sorted(saved_video_for_color)}")

# ─── Write HDF5 v2 schema ───────────────────────────────────────────────────
try:
    commit = subprocess.check_output(['git','rev-parse','--short','HEAD'],
                                      cwd=_REPO_ROOT).decode().strip()
except Exception:
    commit = 'unknown'

print(f"\nWriting {OUT_H5}...")
with h5py.File(OUT_H5, 'w') as f:
    f.attrs['format_version']    = '2.0'
    f.attrs['env_id']            = 'SO101PlaceCube-v1'
    f.attrs['control_mode']      = 'pd_joint_pos'
    f.attrs['n_distractors']     = 0
    f.attrs['use_real_bowl']     = True
    f.attrs['domain_randomization'] = False
    f.attrs['apply_jitter']      = True
    f.attrs['rgb_h']             = IMG_H
    f.attrs['rgb_w']             = IMG_W
    f.attrs['state_dim']         = 21
    f.attrs['action_dim']        = 6
    f.attrs['arm_delta_max']     = 0.05
    f.attrs['grip_delta_max']    = 0.20
    f.attrs['num_demos']         = len(demos)
    f.attrs['num_colors']        = NUM_COLORS
    f.attrs['T']                 = demos[0]['actions'].shape[0] if demos else 0
    f.attrs['reward_v_min']      = -20.0
    f.attrs['reward_v_max']      = 20.0
    f.attrs['collector_commit']  = commit
    f.attrs['collected_at_utc']  = datetime.datetime.utcnow().isoformat() + 'Z'
    f.attrs['collector_deviation_control_mode'] = (
        "Spec called for pd_joint_target_delta_pos but it has a target-accumulation bug "
        "under cube/finger collision. Actions in this file are ABSOLUTE joint targets (rad). "
        "Convert at load time: delta[t] = action[t] - action[t-1] with action[-1]=QPOS_START, "
        "then normalize by [0.05]*5+[0.2]."
    )
    f.attrs['collector_deviation_msvecenv'] = (
        "Spec called for ManiSkillVectorEnv wrapper but A/B test showed it kills grasp "
        "dynamics. Demos collected without it. Each demo is one uninterrupted episode."
    )
    f.attrs['collector_deviation_dr'] = (
        "Spec called for domain_randomization=True. DR drops scripted-IK grasp rate <5%. "
        "Demos collected with DR=False; trainer can apply per-step visual jitter at training time."
    )
    f.attrs['state_layout'] = 'bowl_xyz_robot_frame(3) + target_qpos(6, synthesized=action[t-1]) + goal_color_onehot(6) + noisy_qpos(6)'

    for i, d in enumerate(demos):
        g = f.create_group(f'demo_{i:03d}')
        g.create_dataset('obs/rgb',   data=d['rgb'],     compression='gzip', compression_opts=4)
        g.create_dataset('obs/state', data=d['state'],   compression='gzip', compression_opts=4)
        g.create_dataset('actions',   data=d['actions'], compression='gzip', compression_opts=4)
        g.create_dataset('rewards',   data=d['rewards'])
        g.create_dataset('terminals', data=d['terminals'])
        g.attrs['color_idx']  = d['color_idx']
        g.attrs['cube_pos']   = d['cube_pos']
        g.attrs['bowl_pos']   = d['bowl_pos']
        g.attrs['seed']       = d['seed']
        g.attrs['success']    = True
        g.attrs['return_sum'] = d['return_sum']

sz = os.path.getsize(OUT_H5)/1e6
print(f"Wrote {len(demos)} demos → {OUT_H5}  ({sz:.1f} MB)")

with open(OUT_META,'w') as f:
    json.dump(dict(
        num_demos=len(demos),
        per_color_counts=color_counts,
        seeds_used=[d['seed'] for d in demos],
        state_dim=21, action_dim=6,
        control_mode='pd_joint_pos',
        rgb_resolution=[IMG_H, IMG_W],
        sanity_videos_for_colors=sorted(saved_video_for_color),
        notes='50-demo collection using V3-style env (no MSVecEnv, no DR). '
              'Same v2 schema as 2-demo pipeline test. Deviation flags in HDF5 attrs.',
    ), f, indent=2)
print(f"Meta → {OUT_META}")

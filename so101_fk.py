"""Pure-numpy forward kinematics for the SO101 follower arm.

Mirrors the kinematic chain in envs/robot/so101.urdf so the Linux deploy
script can compute fingertip positions in the robot base frame WITHOUT a
simulator. Used to height-gate the hardcoded gripper close: the cube sits on
the table (top face ~2 cm above z=0, base_link is also at z=0), so "fingertip
2 cm below the cube's top face" is just "fingertip z ~ table level".

Joint order matches infer_linux.JOINT_NAMES:
    [pan, lift, elbow, wrist_flex, wrist_roll, gripper]   (sim radians)

Self-test (compares against ManiSkill SAPIEN link poses):
    python so101_fk.py
"""
import numpy as np


def _rpy(roll, pitch, yaw):
    """URDF fixed-axis rpy -> 3x3 rotation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _T(xyz, rpy):
    M = np.eye(4)
    M[:3, :3] = _rpy(*rpy)
    M[:3, 3] = xyz
    return M


def _Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    M = np.eye(4)
    M[0, 0], M[0, 1] = c, -s
    M[1, 0], M[1, 1] = s, c
    return M


# Fixed joint-origin transforms straight from so101.urdf (parent -> child).
_ORIGIN_PAN        = _T([0.0388353, 0.0, 0.0624],          [3.14159, 0.0, -3.14159])
_ORIGIN_LIFT       = _T([-0.0303992, -0.0182778, -0.0542], [-1.5708, -1.5708, 0.0])
_ORIGIN_ELBOW      = _T([-0.11257, -0.028, 0.0],           [0.0, 0.0, 1.5708])
_ORIGIN_WRISTFLEX  = _T([-0.1349, 0.0052, 0.0],            [0.0, 0.0, -1.5708])
_ORIGIN_WRISTROLL  = _T([0.0, -0.0611, 0.0181],            [1.5708, 0.0486795, 3.14159])
_ORIGIN_GRIPPER    = _T([0.0202, 0.0188, -0.0234],         [1.5708, 0.0, 0.0])
_ORIGIN_FINGER1TIP = _T([-0.002, 0.0, -0.092],             [0.0, 0.0, 0.0])   # on gripper_link
_ORIGIN_FINGER2TIP = _T([-0.01, -0.077, 0.02],             [0.0, 0.0, 0.0])   # on moving jaw


def fk_frames(q):
    """Return base-frame 4x4 transforms for the key links.

    q: length-6 array [pan, lift, elbow, wrist_flex, wrist_roll, gripper] (rad).
    Returns dict with 'gripper_link', 'moving_jaw', 'finger1_tip', 'finger2_tip'.
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    T_shoulder = _ORIGIN_PAN @ _Rz(q[0])
    T_upper    = T_shoulder @ _ORIGIN_LIFT @ _Rz(q[1])
    T_lower    = T_upper @ _ORIGIN_ELBOW @ _Rz(q[2])
    T_wrist    = T_lower @ _ORIGIN_WRISTFLEX @ _Rz(q[3])
    T_gripper  = T_wrist @ _ORIGIN_WRISTROLL @ _Rz(q[4])
    T_jaw      = T_gripper @ _ORIGIN_GRIPPER @ _Rz(q[5])
    return {
        "gripper_link": T_gripper,
        "moving_jaw": T_jaw,
        "finger1_tip": T_gripper @ _ORIGIN_FINGER1TIP,
        "finger2_tip": T_jaw @ _ORIGIN_FINGER2TIP,
    }


def finger_positions(q):
    """(finger1_tip_pos, finger2_tip_pos) in base frame, each (3,)."""
    f = fk_frames(q)
    return f["finger1_tip"][:3, 3].copy(), f["finger2_tip"][:3, 3].copy()


def tcp_pos(q):
    """Tool-center point = fingertip midpoint in base frame (matches sim tcp_pos)."""
    p1, p2 = finger_positions(q)
    return (p1 + p2) / 2.0


def tcp_jacobian(q, eps=1e-6):
    """Numerical 3x5 Jacobian d(tcp_xyz)/d(arm joints). Gripper joint excluded."""
    q = np.asarray(q, dtype=np.float64).flatten()
    J = np.zeros((3, 5))
    base = tcp_pos(q)
    for j in range(5):
        dq = q.copy()
        dq[j] += eps
        J[:, j] = (tcp_pos(dq) - base) / eps
    return J


def nudge_arm_joints(q, delta_xyz, max_joint_step=0.2, iters=4):
    """Length-6 joint delta (gripper entry 0) that moves the TCP by delta_xyz
    (base frame, metres) via a few damped least-squares IK iterations."""
    q0 = np.asarray(q, dtype=np.float64).flatten()
    target = tcp_pos(q0) + np.asarray(delta_xyz, dtype=np.float64)
    q_cur = q0.copy()
    lam = 1e-4
    for _ in range(iters):
        resid = target - tcp_pos(q_cur)
        J = tcp_jacobian(q_cur)
        q_cur[:5] += J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), resid)
    out = np.zeros(6)
    out[:5] = np.clip(q_cur[:5] - q0[:5], -max_joint_step, max_joint_step)
    return out


if __name__ == "__main__":
    # Verify against ManiSkill / SAPIEN link poses for several random configs.
    import gymnasium as gym
    import torch
    import mani_skill.envs  # noqa: F401  (registers envs)
    import envs.place  # noqa: F401  (registers PlaceCube + so101 agent)

    env = gym.make("SO101PlaceCube-v1", num_envs=1)
    env.reset(seed=0)
    base_env = env.unwrapped
    agent = base_env.agent
    robot = agent.robot

    # Robot root pose (base_link) in world: position + yaw. FK is in base frame,
    # so transform FK output by the root pose before comparing to world poses.
    root = robot.pose
    root_p = root.p[0].cpu().numpy()
    root_R = root.to_transformation_matrix()[0, :3, :3].cpu().numpy()

    def to_world(p_base):
        return root_R @ p_base + root_p

    rng = np.random.default_rng(0)
    lower = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.174533])
    upper = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 2.0944])

    max_err = 0.0
    for trial in range(8):
        if trial == 0:
            q = np.array([0.0, -80.791, 36.747, 86.901, -82.154, 120.0]) * np.pi / 180
        else:
            q = rng.uniform(lower, upper)
        full = robot.get_qpos()
        full[0, :6] = torch.tensor(q, dtype=full.dtype)
        robot.set_qpos(full)
        try:
            base_env.scene._gpu_apply_all()
            base_env.scene.px.gpu_update_articulation_kinematics()
            base_env.scene._gpu_fetch_all()
        except (AttributeError, RuntimeError):
            pass  # CPU sim updates link poses immediately on set_qpos

        sim_f1 = agent.finger1_tip.pose.p[0].cpu().numpy()
        sim_f2 = agent.finger2_tip.pose.p[0].cpu().numpy()
        sim_tcp = agent.tcp_pos[0].cpu().numpy()

        f1, f2 = finger_positions(q)
        fk_tcp = to_world(tcp_pos(q))
        e1 = np.linalg.norm(to_world(f1) - sim_f1)
        e2 = np.linalg.norm(to_world(f2) - sim_f2)
        et = np.linalg.norm(fk_tcp - sim_tcp)
        max_err = max(max_err, e1, e2, et)
        print(f"trial {trial}: f1_err={e1*1000:6.2f}mm  f2_err={e2*1000:6.2f}mm  "
              f"tcp_err={et*1000:6.2f}mm  | sim_tcp={sim_tcp}  fk_tcp={fk_tcp}")

    print(f"\nmax error across trials: {max_err*1000:.3f} mm")
    print("FK", "OK" if max_err < 1e-3 else "MISMATCH — check transforms")

    # IK nudge check (pure numpy, no sim): request a 5 mm xy move, verify TCP lands there.
    q_test = np.array([0.0, -80.791, 36.747, 86.901, -82.154, 120.0]) * np.pi / 180
    want = np.array([0.005, -0.003, 0.0])
    dq = nudge_arm_joints(q_test, want)
    got = tcp_pos(q_test + dq) - tcp_pos(q_test)
    ik_err = np.linalg.norm(got - want)
    print(f"IK nudge: want={want*1000} mm  got={got*1000} mm  err={ik_err*1000:.3f} mm  "
          + ("OK" if ik_err < 1e-4 else "CHECK"))
    env.close()

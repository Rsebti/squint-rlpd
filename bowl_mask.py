"""Geometric bowl mask: project the known bowl (a truncated cone on the table)
from the current wrist-camera pose into the image and rasterize its 2-D silhouette.

The bowl is white and its colour overlaps the cubes, so colour-based masking is
flaky. Its 3-D pose is known (taught xy in the robot base frame, sitting on the
table at z=0), and the wrist camera pose follows the gripper via FK — so we can
project a faithful silhouette that's appearance-invariant.

    from bowl_mask import bowl_mask
    m = bowl_mask(q, (0.20, 0.20), (det_h, det_w))   # uint8 0/255, same HxW

Pure numpy + cv2. Tunables are module globals so the projection can be dialled in
by eye in final_utils/mask_live (the camera-frame convention especially).
"""
import numpy as np
import cv2

from so101_fk import fk_frames

# ── Wrist camera mount (mirror of envs/base_random_env.py) ───────────────────
# Camera pose in base frame = gripper_link_pose * local_offset, where local_offset
# is this position + orientation relative to gripper_link.
WRIST_CAMERA_BASE_POS = (-0.0006, 0.0498, -0.0641)
WRIST_CAMERA_BASE_ROT_RAD = (np.deg2rad(-90), np.deg2rad(91), np.deg2rad(-35.31))  # roll,pitch,yaw
WRIST_CAMERA_FOV = np.deg2rad(76.92)        # SAPIEN vertical FOV (fovy)

# ── Bowl geometry (truncated cone, silhouette only) ──────────────────────────
BOWL_R_BOTTOM = 0.050       # bottom radius (10 cm ⌀)
BOWL_R_TOP = 0.075          # top radius   (15 cm ⌀)
BOWL_H = 0.045              # height (4.5 cm)
BOWL_R_MARGIN = 1.10        # inflate radius for calibration slack
BOWL_MASK_DILATE_PX = 2     # extra px around the filled silhouette

# ── SAPIEN camera frame (x fwd, y left, z up) → OpenCV optical (x right, y down,
#    z fwd). This is the main correctness knob; if the projection is mirrored or
#    flipped during eye-tuning, flip the signs here.
S2CV = np.array([[0.0, -1.0, 0.0],
                 [0.0, 0.0, -1.0],
                 [1.0, 0.0, 0.0]])


def _euler_to_R(roll, pitch, yaw):
    """Reproduce the env's euler→quaternion (base_random_env._update_wrist_camera_pose),
    then quaternion→rotation matrix, so the mount orientation matches sim exactly."""
    cj, sj = np.cos(pitch / 2), np.sin(pitch / 2)
    ck, sk = np.cos(yaw / 2), np.sin(yaw / 2)
    ci, si = np.cos(roll / 2), np.sin(roll / 2)
    q_py_w, q_py_x, q_py_y, q_py_z = cj * ck, sj * sk, sj * ck, cj * sk
    qw = q_py_w * ci - q_py_x * si
    qx = q_py_w * si + q_py_x * ci
    qy = q_py_y * ci + q_py_z * si
    qz = q_py_z * ci - q_py_y * si
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _mount_T():
    T = np.eye(4)
    T[:3, :3] = _euler_to_R(*WRIST_CAMERA_BASE_ROT_RAD)
    T[:3, 3] = WRIST_CAMERA_BASE_POS
    return T


def intrinsics(img_w, img_h, fovy=None):
    """Pinhole K from the vertical FOV at the given resolution (square pixels,
    centred principal point — matches the SAPIEN render and the 16:9 calibration)."""
    fovy = WRIST_CAMERA_FOV if fovy is None else fovy
    fy = (img_h / 2.0) / np.tan(fovy / 2.0)
    fx = fy
    return np.array([[fx, 0, img_w / 2.0], [0, fy, img_h / 2.0], [0, 0, 1.0]])


def _cone_points(cx, cy, bowl_z=0.0, n=48):
    """Rim circles of the truncated cone in the base frame (bottom on the table)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rb, rt = BOWL_R_BOTTOM * BOWL_R_MARGIN, BOWL_R_TOP * BOWL_R_MARGIN
    bot = np.stack([cx + rb * np.cos(th), cy + rb * np.sin(th), np.full(n, bowl_z)], axis=1)
    top = np.stack([cx + rt * np.cos(th), cy + rt * np.sin(th), np.full(n, bowl_z + BOWL_H)], axis=1)
    return np.vstack([bot, top])


def camera_pose_base(q):
    """4×4 wrist-camera pose in the base frame (SAPIEN convention) from joint q."""
    return fk_frames(q)["gripper_link"] @ _mount_T()


def bowl_mask(q, bowl_xy, img_hw, bowl_z=0.0, fovy=None, dilate_px=None):
    """uint8 0/255 silhouette of the bowl in an `img_hw`=(H,W) image, given joint
    angles q (sim radians, 6-vec) and the bowl centre xy in the base frame.
    Returns an all-zero mask if the bowl projects fully behind the camera."""
    h, w = img_hw[0], img_hw[1]
    mask = np.zeros((h, w), np.uint8)

    T_cam_base = np.linalg.inv(camera_pose_base(q))
    pts = _cone_points(bowl_xy[0], bowl_xy[1], bowl_z)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    p_cam = (T_cam_base @ pts_h.T).T[:, :3]        # SAPIEN camera frame
    p_cv = (S2CV @ p_cam.T).T                      # OpenCV optical frame

    front = p_cv[:, 2] > 1e-4                      # in front of the camera
    if front.sum() < 3:
        return mask
    K = intrinsics(w, h, fovy)
    proj = (K @ p_cv[front].T).T
    uv = (proj[:, :2] / proj[:, 2:3])
    uv = np.round(uv).astype(np.int32)

    hull = cv2.convexHull(uv)
    cv2.fillConvexPoly(mask, hull, 255)
    d = BOWL_MASK_DILATE_PX if dilate_px is None else dilate_px
    if d > 0:
        mask = cv2.dilate(mask, np.ones((2 * d + 1, 2 * d + 1), np.uint8))
    return mask

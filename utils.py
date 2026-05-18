import time

import numpy as np
import cv2
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from mani_skill.utils import common
from mani_skill.utils.visualization.misc import images_to_video, tile_images, put_info_on_image
from mani_skill.utils.wrappers.record import RecordEpisode

# ---------------------------  Wrappers --------------------------------------#

class DownsampleObsWrapper(gym.ObservationWrapper):
    """Downsamples RGB observations from render size to target size using area interpolation.

    `target_size` may be an int (square output HxH) or a tuple (H, W).
    Expects input in (B, H, W, C) format.
    """
    def __init__(self, env, target_size):
        super().__init__(env)
        # Normalize to (H, W) tuple.
        if isinstance(target_size, int):
            self.target_h, self.target_w = target_size, target_size
        else:
            self.target_h, self.target_w = int(target_size[0]), int(target_size[1])
        self.target_size = (self.target_h, self.target_w)
        old_rgb_space = self.observation_space['rgb']
        C = old_rgb_space.shape[-1]
        self.observation_space['rgb'] = gym.spaces.Box(
            low=0, high=255, shape=(self.target_h, self.target_w, C), dtype=old_rgb_space.dtype
        )

    def observation(self, obs):
        rgb = obs['rgb']  # (B, H, W, C) or (H, W, C)
        if rgb.shape[-3] == self.target_h and rgb.shape[-2] == self.target_w:
            return obs  # Already at target size

        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        rgb = rgb.permute(0, 3, 1, 2)
        rgb = F.interpolate(rgb.float(), size=(self.target_h, self.target_w), mode='area').to(torch.uint8)
        rgb = rgb.permute(0, 2, 3, 1)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs



class ColorJitterWrapper(gym.ObservationWrapper):
    """Applies random color jitter to RGB observations for sim2real robustness.

    Expects input in (B, H, W, C) format.
    """
    def __init__(self, env, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
        super().__init__(env)
        self.jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def observation(self, obs):
        rgb = obs['rgb']  # (B, H, W, C) or (H, W, C) uint8

        # Handle batched and unbatched cases
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        # (B, H, W, C) -> (B, C, H, W) for ColorJitter
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = self.jitter(rgb.float() / 255.0)
        # (B, C, H, W) -> (B, H, W, C)
        rgb = rgb.permute(0, 2, 3, 1)

        # Back to uint8
        rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs


class GoalColorOverlayWrapper(gym.Wrapper):
    """Stamps the current goal color in the top-left of each per-env render frame.

    Insert between the vectorized env and RecordEpisode so the saved MP4s show
    which color the policy was conditioned on for each parallel env. Affects only
    ``render()``; obs/step/reward are untouched.
    """

    COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange"]
    # Text colors in RGB (matches the env's rendered RGB frames).
    TEXT_RGB = [
        (255, 0, 0),    # red
        (0, 80, 255),   # blue
        (0, 255, 0),    # green
        (255, 255, 0),  # yellow
        (180, 0, 220),  # purple
        (255, 140, 0),  # orange
    ]

    def render(self):
        frames = self.env.render()
        unwrapped = self.env.unwrapped
        if not hasattr(unwrapped, "goal_color_idx"):
            return frames

        is_tensor = isinstance(frames, torch.Tensor)
        np_frames = frames.cpu().numpy() if is_tensor else np.asarray(frames)
        # Render output is (N, H, W, 3) uint8 for vector envs, (H, W, 3) for single env.
        if np_frames.ndim == 4:
            goal_arr = unwrapped.goal_color_idx.tolist()
            for i in range(np_frames.shape[0]):
                self._stamp_frame(np_frames[i], int(goal_arr[i]))
        elif np_frames.ndim == 3:
            idx_t = unwrapped.goal_color_idx
            idx = int(idx_t[0].item()) if idx_t.ndim > 0 else int(idx_t.item())
            self._stamp_frame(np_frames, idx)

        if is_tensor:
            return torch.from_numpy(np_frames).to(frames.device)
        return np_frames

    @classmethod
    def _stamp_frame(cls, frame: np.ndarray, color_idx: int):
        """In-place: draw 'goal: <name>' on the frame with a black outline."""
        h, w = frame.shape[:2]
        # Font scale that stays legible at 128x128 and 512x512 alike.
        scale = max(0.35, w / 320.0)
        thickness = 1 if w <= 256 else 2
        outline = thickness + 2
        text = f"goal: {cls.COLOR_NAMES[color_idx]}"
        org = (5, max(12, int(14 * scale)))
        # Outline first (black), then colored fill on top.
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), outline, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    cls.TEXT_RGB[color_idx], thickness, cv2.LINE_AA)


class ClockedRecordEpisode(RecordEpisode):
    """RecordEpisode with a single wall/sim clock stamped on each video frame.

    Also caps the per-frame env tiling at MAX_ENVS_IN_VIDEO so the .mp4 stays
    readable as NUM_EVAL_ENVS grows past ~64. Beyond that the tile cells
    become too small to interpret and the file grows linearly in env count.
    The cap takes the first MAX_ENVS_IN_VIDEO envs (DR makes them visually
    diverse anyway). Best/worst selection would need a per-env frame buffer.

    The clock overlay is added AFTER per-env tiling, so there is exactly one
    clock per .mp4 frame. Resets each time a video is flushed.

    Shows: wall-clock seconds since the video started, sim seconds (= step *
    control_timestep), and the ratio sim/wall. >1 means sim runs faster than
    real-time; <1 means slower.
    """

    MAX_ENVS_IN_VIDEO = 30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._control_dt = float(self.base_env.control_timestep)
        self._clock_wall_start = None
        self._clock_sim_steps = 0

    def capture_image(self, infos=None):
        # Override RecordEpisode.capture_image to cap the number of envs we
        # tile into the video before composing the mosaic.
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]
        if len(img) > self.MAX_ENVS_IN_VIDEO:
            img = img[: self.MAX_ENVS_IN_VIDEO]
        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                # Recompute nrows for the (capped) env count. RecordEpisode's
                # self.video_nrows = sqrt(num_eval_envs) which is set for the
                # FULL eval-envs count (e.g. 16 for 256 envs); applying that
                # to a 30-env cap gives 16 rows × 2 cols = vertical strip.
                # Use a near-square layout for the capped count instead.
                n = len(img)
                nrows = max(1, int(np.floor(np.sqrt(n))))
                img = tile_images(img, nrows=nrows)
        return self._stamp_clock(img)

    def _stamp_clock(self, img: np.ndarray) -> np.ndarray:
        if self._clock_wall_start is None:
            self._clock_wall_start = time.perf_counter()
            wall_s = 0.0
        else:
            self._clock_sim_steps += 1
            wall_s = time.perf_counter() - self._clock_wall_start
        sim_s = self._clock_sim_steps * self._control_dt
        ratio = (sim_s / wall_s) if wall_s > 0.05 else 1.0
        text = f"wall {wall_s:5.2f}s  sim {sim_s:5.2f}s  x{ratio:4.2f}"
        h, w = img.shape[:2]
        scale = max(0.5, w / 900.0)
        thickness = max(1, int(round(scale * 1.6)))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        org = (w - tw - 10, h - 10)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 0), thickness, cv2.LINE_AA)
        return img

    def flush_video(self, *args, **kwargs):
        ret = super().flush_video(*args, **kwargs)
        self._clock_wall_start = None
        self._clock_sim_steps = 0
        return ret


# ---------------------------  Extra Utils --------------------------------------#

def calc_buffer_memory(rgb_dim, state_dim, action_dim, max_length, rgb_dtype=np.uint8, store_next_obs=True):
    """Calculate memory required for buffer in GB and print it.

    Args:
        rgb_dim: Flattened dimension of rgb observation 
        state_dim: Dimension of state observation
        action_dim: Dimension of action space
        max_length: Maximum buffer length
        rgb_dtype: Data type for rgb storage 
        store_next_obs: Whether buffer stores next_obs separately (2x memory for obs)
    """
    obs_multiplier = 2 if store_next_obs else 1

    rgb_bytes = max_length * rgb_dim * np.dtype(rgb_dtype).itemsize * obs_multiplier
    state_bytes = max_length * state_dim * np.dtype(np.float32).itemsize * obs_multiplier
    act_bytes = max_length * action_dim * np.dtype(np.float32).itemsize
    other_bytes = max_length * np.dtype(np.float32).itemsize * 3

    # Total memory in GB
    total_gb = (rgb_bytes + state_bytes + act_bytes + other_bytes) / (1024**3)

    return total_gb






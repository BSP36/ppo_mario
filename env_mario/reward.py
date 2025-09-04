"""Reward function definitions for reinforcement learning in Super Mario Bros.

This module provides custom wrappers and frame processing utilities to shape rewards and observations
for training reinforcement learning agents in the Super Mario Bros environment.
"""
import cv2
import numpy as np
from gym import Wrapper
from gym.spaces import Box


def process_frame(frame, num_color: int, frame_size: tuple):
    """
    Processes a single environment frame for use in reinforcement learning.

    This function resizes the input frame to the specified frame_size, converts it to grayscale
    if num_color is 1, and normalizes pixel values to [0, 1]. The output shape is
    (num_color, height, width).

    Args:
        frame (np.ndarray): The input frame from the environment.
        num_color (int): Number of color channels (1 for grayscale, 3 for RGB).
        frame_size (tuple): Desired output frame size as (height, width).

    Returns:
        np.ndarray: The processed frame with shape (num_color, height, width).
    """
    if frame is not None:
        # Resize frame to the target size
        frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
        if num_color == 1:
            # Convert to grayscale and add channel dimension
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[None, :, :]
        else:
            # Transpose to channel-first format
            frame = np.transpose(frame, (2, 0, 1))
        return frame.astype(np.float32) / 255.0
    else:
        return np.zeros((num_color, frame_size[0], frame_size[1]), dtype=np.float32)
        

class MarioDenseReward(Wrapper):
    """
    Dense, scale-aware reward wrapper for Super Mario Bros environments.

    This class shapes the reward signal for reinforcement learning agents by combining multiple components:
      - Progress: Smooth reward for forward movement (Δx), scaled and saturated.
      - Backward/Idle: Penalties for moving backward or idling.
      - Time: Small penalty per time tick to encourage faster completion.
      - Score: Shaping for score increments (e.g., coins, enemies).
      - Powerup/Damage: Bonuses for gaining powerups, penalties for losing them.
      - Stuck: Penalty if Mario is stuck for several steps.
      - Terminal: Bonus for finishing the level, penalty for death or timeout.
      - Reward is clipped to a stable range.

    Args:
        env (gym.Env): The Super Mario Bros environment to wrap.
        world (int): World number for tracking progress.
        stage (int): Stage number for tracking progress.
        screen (object, optional): Optional screen recorder.
        num_color (int, optional): Number of color channels in processed frames (1=grayscale, 3=RGB). Default: 3.
        frame_size (tuple, optional): Output frame size as (height, width). Default: (32, 32).
        k_progress (float, optional): Scale for progress reward. Default: 0.10.
        dx_scale (float, optional): Saturation scale for progress (pixels). Default: 5.0.
        k_backward (float, optional): Penalty for moving backward. Default: 0.05.
        k_idle (float, optional): Penalty for idle steps. Default: 0.01.
        k_time (float, optional): Penalty per time tick. Default: 0.01.
        k_score (float, optional): Reward shaping for score increments. Default: 0.05.
        k_powerup (float, optional): Bonus for gaining powerups. Default: 1.0.
        k_hurt (float, optional): Penalty for losing powerups. Default: 2.0.
        stuck_patience (int, optional): Steps allowed without progress before stuck penalty. Default: 20.
        stuck_dx_thresh (float, optional): Threshold for considering Mario stuck (pixels). Default: 0.5.
        k_stuck (float, optional): Penalty when stuck. Default: 0.5.
        finish_bonus (float, optional): Bonus for finishing the level. Default: 40.0.
        death_penalty (float, optional): Penalty for dying or failing the level. Default: 15.0.
        clip_abs (float, optional): Absolute value to clip the reward. Default: 5.0.
    """
    def __init__(
        self,
        env,
        world: int,
        stage: int,
        screen: object = None,
        num_color: int = 3,
        frame_size: tuple = (32, 32),
        k_progress: float = 0.10,   # scale for progress via tanh
        dx_scale: float = 5.0,      # progress saturation scale (pixels)
        k_backward: float = 0.05,   # per-step penalty when moving backward
        k_idle: float = 0.01,       # small penalty when Δx ~ 0
        k_time: float = 0.01,       # penalty per elapsed time tick
        k_score: float = 0.05,      # shaping for (Δscore / 100)
        k_powerup: float = 1.0,     # bonus per status level gained
        k_hurt: float = 2.0,        # penalty per status level lost
        stuck_patience: int = 20,   # steps allowed without noticeable progress
        stuck_dx_thresh: float = 0.5,
        k_stuck: float = 0.5,       # penalty when considered stuck
        finish_bonus: float = 40.0,
        death_penalty: float = 15.0,
        clip_abs: float = 5.0,
    ):

        super().__init__(env)
        self.observation_space = Box(
            low=0, high=255, shape=(num_color, frame_size[0], frame_size[1]), dtype=np.uint8
        )
        self.num_color = num_color
        self.frame_size = frame_size

        # Environment meta
        self.world = world
        self.stage = stage
        self.screen = screen

        # Internal trackers
        self.last_x = 40
        self.last_score = 0
        self.last_time = 400
        self.last_status = 0  # 0=small,1=tall,2=fireball (env usually uses this scale)
        self.no_progress_steps = 0

        # Hyperparameters
        self.k_progress = k_progress
        self.dx_scale = dx_scale
        self.k_backward = k_backward
        self.k_idle = k_idle
        self.k_time = k_time
        self.k_score = k_score
        self.k_powerup = k_powerup
        self.k_hurt = k_hurt
        self.stuck_patience = stuck_patience
        self.stuck_dx_thresh = stuck_dx_thresh
        self.k_stuck = k_stuck
        self.finish_bonus = finish_bonus
        self.death_penalty = death_penalty
        self.clip_abs = clip_abs

    def _status_to_int(self, status):
        if status is None:
            return self.last_status
        if isinstance(status, (int, np.integer)):
            return int(status)
        mapping = {"small": 0, "tall": 1, "fireball": 2}
        return mapping.get(status, self.last_status)

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        if self.screen:
            self.screen.record(obs)
        obs = process_frame(obs, self.num_color, self.frame_size)

        # Read info safely
        x = info.get("x_pos", self.last_x)
        time_left = info.get("time", self.last_time)
        score = info.get("score", self.last_score)
        status = self._status_to_int(info.get("status"))

        # Deltas
        dx = float(x - self.last_x)
        dtime = max(0, int(self.last_time) - int(time_left))  # number of ticks elapsed
        dscore = int(score) - int(self.last_score)
        dstatus = status - self.last_status

        # Smooth progress reward
        r_progress = self.k_progress * np.tanh(dx / self.dx_scale)

        # Backward / idle small penalties
        r_bwd = -self.k_backward if dx < -1e-6 else 0.0
        r_idle = -self.k_idle if abs(dx) < 1e-6 else 0.0

        # Small time penalty per tick
        r_time = -self.k_time * dtime

        # Score shaping (score often increments in 100s)
        r_score = self.k_score * (dscore / 100.0)

        # Powerup/damage shaping
        r_status = 0.0
        if dstatus > 0:
            r_status += self.k_powerup * dstatus
        elif dstatus < 0:
            r_status -= self.k_hurt * abs(dstatus)

        # Stuck penalty if no progress for several steps
        if abs(dx) <= self.stuck_dx_thresh:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        r_stuck = 0.0
        if self.no_progress_steps >= self.stuck_patience:
            r_stuck = -self.k_stuck
            self.no_progress_steps = 0  # reset after penalizing

        # Sum dense reward
        reward = r_progress + r_bwd + r_idle + r_time + r_score + r_status + r_stuck

        # Terminal bonuses/penalties
        if done:
            if info.get("flag_get", False):
                reward += self.finish_bonus
            else:
                # Death, pit, or timeout
                reward -= self.death_penalty

        # Clip to keep magnitudes stable
        reward = float(np.clip(reward, -self.clip_abs, self.clip_abs))

        # Update trackers
        self.last_x = x
        self.last_time = time_left
        self.last_score = score
        self.last_status = status

        return obs, reward, done, info

    def reset(self):
        """
        Reset the environment and internal state variables.

        Returns:
            np.ndarray: The processed initial observation.
        """
        self.last_x = 40
        self.last_time = 400
        self.last_score = 0
        self.last_status = 0
        self.no_progress_steps = 0

        obs = self.env.reset()
        if self.screen:
            self.screen.record(obs)
        return process_frame(obs, self.num_color, self.frame_size)

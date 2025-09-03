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


class CustomReward(Wrapper):
    """
    Custom reward wrapper for Super Mario Bros environment.

    This wrapper modifies the reward signal to encourage progress, penalize time loss,
    and provide bonuses for level completion. It also processes the observation frames.

    Args:
        env (gym.Env): The environment to wrap.
        world (int): The world number in the game.
        stage (int): The stage number in the game.
        screen (object, optional): Optional screen object for recording states.
        num_color (int, optional): Number of color channels in the observation. Defaults to 3.
        frame_size (tuple, optional): Size (height, width) to resize frames to. Defaults to (32, 32).
    """
    def __init__(
        self,
        env,
        world: int,
        stage: int,
        screen: object = None,
        num_color: int = 3,
        frame_size: tuple = (32, 32)
    ):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(num_color, frame_size[0], frame_size[1]))
        self.num_color = num_color
        self.frame_size = frame_size
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage
        self.screen = screen
        self.time = 400

    def step(self, action):
        """
        Take an action in the environment, process the observation, and compute a custom reward.

        The reward is shaped as follows:
        - Reward for score increase.
        - Penalty for time decrement.
        - Reward for moving forward, penalty for moving backward or standing still.
        - Bonus for level completion, penalty for failure.

        Args:
            action: The action to perform.

        Returns:
            tuple: (processed_state, shaped_reward, done, info)
        """
        state, reward, done, info = self.env.step(action)
        if self.screen:
            self.screen.record(state)
        state = process_frame(state, self.num_color, self.frame_size)

        # Reward for score increase
        reward += (info["score"] - self.curr_score) / 40.0
        self.curr_score = info["score"]

        # Penalize for time decrement
        if info["time"] != self.time:
            reward -= 1
        self.time = info["time"]

        # Reward for moving forward, penalize for moving backward or standing still
        if self.current_x < info["x_pos"]:
            reward += 1
        elif self.current_x > info["x_pos"]:
            reward -= 1
        else:
            reward -= 0.1

        # Bonus for level completion, penalty for failure
        if done:
            if info.get("flag_get", False):
                reward += 200
            else:
                reward -= 50

        self.current_x = info["x_pos"]
        return state, reward / 10.0, done, info

    def reset(self):
        """
        Reset the environment and internal state variables.

        Returns:
            np.ndarray: The processed initial observation.
        """
        self.curr_score = 0
        self.current_x = 40
        self.time = 400
        return process_frame(self.env.reset(), self.num_color, self.frame_size)
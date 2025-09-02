"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import cv2
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from .reward import CustomReward, CustomSkipFrame


class Screen:
    """
    Utility class for recording gameplay frames and saving them as a video file.

    Args:
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        saved_path (str): Path to save the output video file.
        fps (int, optional): Frames per second for the output video. Defaults to 60.
    """
    def __init__(self, width: int, height: int, saved_path: str, fps: int = 60):
        self.width = width
        self.height = height
        self.saved_path = saved_path
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(saved_path, self.fourcc, self.fps, (width, height))

    def record(self, image_array: np.ndarray):
        """
        Record a single frame to the video file.

        Args:
            image_array (np.ndarray): Frame image in RGB format.
        """
        bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        self.out.write(bgr_image)

    def close(self):
        """
        Release the video writer and finalize the video file.
        """
        self.out.release()


def create_train_env(
    world: int,
    stage: int,
    actions,
    num_colors: int,
    frame_size: tuple,
    num_skip: int,
    output_path: str = None
):
    """
    Create and configure a Super Mario Bros training environment with custom reward and frame skipping.

    Args:
        world (int): World number (e.g., 1 for World 1).
        stage (int): Stage number within the world (e.g., 1 for Stage 1).
        actions (list): List of allowed action sets (e.g., RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT).
        num_colors (int): Number of color channels for the observation (e.g., 1 for grayscale, 3 for RGB).
        frame_size (tuple): Size to which each frame is resized.
        num_skip (int): Number of frames to skip (repeat the same action).
        output_path (str, optional): Path to save gameplay video. If None, video is not recorded.

    Returns:
        gym.Env: Configured Super Mario Bros environment with custom reward and frame skipping.
    """
    print(f"SuperMarioBros-{world}-{stage}-v3")
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v3")
    screen = Screen(256, 240, output_path) if output_path else None

    env = JoypadSpace(env, actions)
    env = CustomReward(env, world, stage, screen)
    env = CustomSkipFrame(env, num_colors, frame_size, num_skip)
    return env


class MarioEnvironment:
    """
    Wrapper class for the Super Mario Bros environment tailored for reinforcement learning.

    Args:
        world (int): The world number in Super Mario Bros (e.g., 1 for World 1).
        stage (int): The stage number within the world (e.g., 1 for Stage 1).
        action_type (str): The type of action set to use. Options are:
            - "right": Only right movement actions.
            - "simple": Simple movement actions.
            - "complex": Full set of complex actions.
        output_path (str, optional): Path to save gameplay video. If None, no video is saved.
    """
    def __init__(self, world: int, stage: int, action_type: str, output_path: str = None):
        # Select action set based on action_type
        if action_type == "right":
            actions = RIGHT_ONLY
        elif action_type == "simple":
            actions = SIMPLE_MOVEMENT
        elif action_type == "complex":
            actions = COMPLEX_MOVEMENT
        else:
            raise NotImplementedError

        # Create the Mario environment
        self.env = create_train_env(
            world,
            stage,
            actions,
            num_colors=3,
            frame_size=(32, 32),
            num_skip=4,
            output_path=output_path
        )
        obs_shape = self.env.observation_space.shape
        self.num_states = obs_shape[0]
        self.height = obs_shape[1]
        self.width = obs_shape[2]
        self.num_actions = len(actions)

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple: (observation, reward, done, info)
        """
        return self.env.step(action)

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            observation: The initial observation after reset.
        """
        return self.env.reset()

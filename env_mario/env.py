import cv2
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym import Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace

from .reward import CustomReward

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


class CustomSkipFrame(Wrapper):
    """
    Wrapper to skip frames and stack the last N processed frames as a single observation.

    This wrapper executes the same action for a specified number of frames (num_skip),
    accumulates the rewards, and stacks the resulting frames along the channel dimension.
    Useful for temporal context in reinforcement learning.

    Args:
        env (gym.Env): The environment to wrap.
        num_color (int): Number of color channels in the observation.
        frame_size (tuple): Size (height, width) to resize frames to.
        num_skip (int, optional): Number of frames to skip and stack. Defaults to 4.
    """
    def __init__(self, env, num_color: int, frame_size: tuple, num_skip: int = 4):
        super(CustomSkipFrame, self).__init__(env)
        self.num_skip = num_skip
        state_shape = (num_skip * num_color, frame_size[0], frame_size[1])
        self.observation_space = Box(low=0, high=255, shape=state_shape)
        self.states = np.zeros(state_shape, dtype=np.float32)

    def step(self, action):
        """
        Repeats the given action for num_skip frames, accumulates the reward,
        and stacks the resulting frames.

        Args:
            action: The action to perform.

        Returns:
            tuple: (stacked_states, total_reward, done, info)
        """
        total_reward = 0
        states = []
        done = False
        info = None
        for _ in range(self.num_skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        self.states = np.concatenate(states, axis=0)
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        """
        Resets the environment and stacks the initial frame num_skip times.

        Returns:
            np.ndarray: The stacked initial observation.
        """
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.num_skip)], axis=0)
        return self.states[None, :, :, :].astype(np.float32)

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
        num_colors (int, optional): Number of color channels for observations (1 for grayscale, 3 for RGB). Defaults to 3.
        frame_size (int, optional): Size (height and width) to which each frame is resized. Defaults to 32.
        num_skip (int, optional): Number of frames to skip (repeat the same action). Defaults to 4.
        output_path (str, optional): Path to save gameplay video. If None, no video is saved. Defaults to None.
    """
    def __init__(
            self,
            world: int,
            stage: int,
            action_type: str,
            num_colors: int = 3,
            frame_size: int = 32,
            num_skip: int = 4,
            output_path: str = None
        ): 
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
        print(f"SuperMarioBros-{world}-{stage}-v3")
        env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v3")
        screen = Screen(256, 240, output_path) if output_path else None
        env = JoypadSpace(env, actions)
        env = CustomReward(env, world, stage, screen)
        self.env = CustomSkipFrame(env, num_colors, (frame_size, frame_size), num_skip)

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

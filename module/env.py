"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import gym_super_mario_bros
from gym.spaces import Box
import gym
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np

# FRAME_X = FRAME_Y = 128
NUM_COLOR = 3
# NUM_COLOR = 1
FRAME_X = FRAME_Y = 32
# FRAME_X = FRAME_Y = 96


class Monitor:
    def __init__(self, width, height, saved_path):
        self.width = width
        self.height = height
        self.saved_path = saved_path
        self.fps = 60
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID', 'MJPG' も可
        self.out = cv2.VideoWriter(saved_path, self.fourcc, self.fps, (width, height))

    def record(self, image_array):
        bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        self.out.write(bgr_image)

    def close(self):
        self.out.release()


def process_frame(frame):
    if frame is not None:
        factor = frame.shape[1] // FRAME_X
        H = frame.shape[0] // factor
        frame = cv2.resize(frame, (FRAME_X, H))# / 255.0
        frame = cv2.copyMakeBorder(frame, FRAME_X - H, 0, 0, 0, cv2.BORDER_REPLICATE)
        if NUM_COLOR == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[None, :, :]
            # import matplotlib.pyplot as plt
            # plt.imshow(frame[0, :,:], cmap="gray")
            # plt.show()
            # print(frame.shape)
            # exit()
        else:            
            # print(frame.shape)
            # import matplotlib.pyplot as plt
            # plt.imshow(frame)
            # plt.show()
            # exit()
            frame = np.transpose(frame, (2, 0, 1))
        return frame / 255.0
    else:
        return np.zeros((NUM_COLOR, FRAME_X, FRAME_Y))


class CustomReward(Wrapper):
    def __init__(self, env=None, world=None, stage=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(NUM_COLOR, FRAME_X, FRAME_Y))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage
        self.monitor = monitor if monitor else None
        self.time = 400

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        # if info["time"] != self.time:
        #     reward -= 0.25
        #     print("time", info["time"], self.time)
        reward -= (info["time"] != self.time)
        self.time = info["time"]
        if self.current_x < info["x_pos"]:
            reward += 1
        elif self.current_x > info["x_pos"]:
            reward -= 1
        else:
            reward -= 0.1

        if done:
            if info["flag_get"]:
                reward += 200
            else:
                reward -= 50

        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        self.time = 400
        return process_frame(self.env.reset())

class CustomSkipFrame(Wrapper):
    def __init__(self, env, num_skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.num_skip = num_skip
        self.observation_space = Box(low=0, high=255, shape=(num_skip * NUM_COLOR, FRAME_X, FRAME_Y))
        self.states = np.zeros((num_skip * NUM_COLOR, FRAME_X, FRAME_Y), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        states = []
        for _ in range(self.num_skip):
            state, reward, done, info = self.env.step(action)
            # print(reward, self.env.time, info)
            # print(action)
            total_reward += reward
            states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        # import matplotlib.pyplot as plt
        # plt.imshow(np.transpose(state, [1,2,0]))
        # plt.axis('off')  # 軸を非表示
        # plt.show()
        # exit()
        self.states = np.concatenate(states, axis=0)
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.num_skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


def create_train_env(world, stage, actions, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v3".format(world, stage))
    # print(env.observation_space.shape)
    # exit()
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None
    # env = Downsample(env, ratio=4)
    # print(env.observation_space.shape)
    # exit()

    env = JoypadSpace(env, actions)
    env = CustomReward(env, world, stage, monitor)
    env = CustomSkipFrame(env)
    return env


class MarioEnvironment:
    def __init__(self, world, stage, action_type, output_path=None):
        if action_type == "right":
            actions = RIGHT_ONLY
        elif action_type == "simple":
            actions = SIMPLE_MOVEMENT
        else:
            actions = COMPLEX_MOVEMENT

        # 1つの環境だけを管理
        self.env = create_train_env(world, stage, actions, output_path=output_path)
        self.num_states = self.env.observation_space.shape[0]
        self.heighth = self.env.observation_space.shape[1]
        self.width = self.env.observation_space.shape[2]
        self.num_actions = len(actions)

    def step(self, action):
        """ 環境でステップを実行し、次の状態・報酬・終了フラグを返す """
        return self.env.step(action)

    def reset(self):
        """ 環境をリセットし、初期状態を返す """
        return self.env.reset()

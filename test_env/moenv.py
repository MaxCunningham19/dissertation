import math
import numpy as np
import gym
from gym.utils import seeding
from PIL import Image
import random
from gym.spaces import Box, Discrete, Dict

ACTION_ARRAY = [(0, -1), (0, 1), (-1, 0), (1, 0)]
SPEED_IDX = 0
GOALS_IDX = 1
BOUNDARY_IDX = 2


class TestMOEnv(gym.Env):
    """
    A MO-enviroment with 3 stationary goals, an exit, the agent has to collect at least one goal before exiting the stage
    Rewards are: Speed to Exit, Goals Completed
    """

    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 3}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, render_mode="human", width=3, height=3, fps=3, max_steps=200):
        self.render_mode = render_mode
        self.frames_per_second = fps
        self.viewer = None
        self.seed()
        self.width = min(width, 3)
        self.height = min(height, 3)
        self.max_steps = max_steps

        self.goal_indexs = [(0, self.height - 1), (self.width - 1, 0), (self.width - 1, self.height - 1)]
        self.exit = (self.width // 2, self.height // 2)
        self.steps = 0
        self.agent = (0, 0)

        ### speed, staying in bounds, goals collected
        self.reward_space_info = ["speed", "goals", "boarders"]
        self.reward_space = Box(low=np.array([-float("inf"), 0.0, -float("inf")]), high=np.array([0.0, 1.0, 0.0]), shape=(3,), dtype=np.float32)
        self.observation_space = Box(low=0, high=max(self.width, self.height) - 1, shape=(2,), dtype=np.int32)
        self.action_space = Discrete(4)  # up, down, left, right = 0, 1, 2, 3

    def step(self, action):
        reward = [-0.1, 0.0, 0.0]
        if self.steps >= self.max_steps:
            return self.get_obs(), reward, False, True, {}

        self.steps += 1
        action_move = ACTION_ARRAY[action]
        new_location = (self.agent[0] + action_move[0], self.agent[1] + action_move[1])
        done = False
        if new_location[0] >= 0 and new_location[0] < self.width and new_location[1] >= 0 and new_location[1] < self.height:
            self.agent = new_location
        else:
            reward[BOUNDARY_IDX] = -1.0

        for i, goal in enumerate(self.goal_indexs):
            if self.agent == goal:
                reward[GOALS_IDX] = 10.0
                self.goal_indexs.pop(i)
                break

        if self.agent == self.exit:
            done = True

        return self.get_obs(), reward, done, False, {}

    def get_obs(self):
        return np.array([self.agent[0], self.agent[1]], dtype=np.int32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.goal_indexs = [(0, self.height - 1), (self.width - 1, 0), (self.width - 1, self.height - 1)]
        self.exit = (self.width // 2, self.height // 2)
        self.agent = (0, 0)
        self.steps = 0

        return self.get_obs(), {}

    def get_image(self):
        matrix = [[(255, 255, 255) for _ in range(self.width)] for _ in range(self.height)]
        matrix[self.agent[1]][self.agent[0]] = (0, 0, 0)
        for goal in self.goal_indexs:
            matrix[goal[1]][goal[0]] = (0, 180, 0)
        matrix[self.exit[1]][self.exit[0]] = (0, 0, 180)
        return matrix

    def render(self, max_width=500):
        img = self.get_image()
        img = np.asarray(img).astype(np.uint8)
        img_height, img_width = img.shape[:2]
        ratio = max_width / img_width
        img = Image.fromarray(img).resize([int(ratio * img_width), int(ratio * img_height)], Image.NEAREST)
        img = np.asarray(img)
        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

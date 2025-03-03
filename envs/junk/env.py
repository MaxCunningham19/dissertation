import math
import numpy as np
import gym
from gym.utils import seeding
from PIL import Image
import random
from gym.spaces import Box, Discrete, Dict

ACTION_ARRAY = [(0, -1), (0, 1), (-1, 0), (1, 0)]


class TestEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 3}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, render_mode="human", width=5, height=1, fps=3, max_steps=200, goal_index=None, agent_index=None):
        self.render_mode = render_mode
        self.frames_per_second = fps
        self.viewer = None
        self.seed()
        self.width = width
        self.height = height
        self.goal_index = goal_index
        if self.goal_index is None:
            self.goal = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        else:
            self.goal = self.goal_index
        self.agent = self.goal
        self.max_steps = max_steps
        self.steps = 0
        self.agent_index = agent_index
        if self.agent_index is None:
            self.agent = self.goal
            while self.agent == self.goal:
                self.agent = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        else:
            self.agent = self.agent_index
        self.observation_space = Box(low=0, high=max(self.width, self.height) - 1, shape=(2,), dtype=np.int32)
        self.action_space = Discrete(4)  # up, down, left, right = 0, 1, 2, 3

    def step(self, action):
        if self.steps >= self.max_steps:
            return self.get_obs(self.agent, self.goal), 0, False, True, {}

        self.steps += 1
        action_move = ACTION_ARRAY[action]
        new_location = (self.agent[0] + action_move[0], self.agent[1] + action_move[1])
        reward = 0
        done = False
        if new_location[0] >= 0 and new_location[0] < self.width and new_location[1] >= 0 and new_location[1] < self.height:
            self.agent = new_location
        else:
            reward = -10

        x_dist = self.goal[0] - self.agent[0]
        y_dist = self.goal[1] - self.agent[1]
        dist = math.sqrt(x_dist * x_dist + y_dist * y_dist)
        reward = reward - dist

        if self.agent == self.goal:
            reward = 100
            done = True

        return self.get_obs(self.agent, self.goal), reward, done, False, {}

    def get_obs(self, agent, goal):
        return np.array([agent[0], agent[1]], dtype=np.int32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.goal_index is None:
            self.goal = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        else:
            self.goal = self.goal_index
        self.steps = 0
        if self.agent_index is None:
            self.agent = self.goal
            while self.agent == self.goal:
                self.agent = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        else:
            self.agent = self.agent_index

        return self.get_obs(self.agent, self.goal), {}

    def get_image(self):
        matrix = [[(255, 255, 255) for _ in range(self.width)] for _ in range(self.height)]
        matrix[self.agent[1]][self.agent[0]] = (0, 0, 0)
        matrix[self.goal[1]][self.goal[0]] = (0, 180, 0)
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

import numpy as np
import gymnasium as gym
from PIL import Image
from gym.spaces import Box, Discrete
from gymnasium.utils import EzPickle

COST_NOOP = 0.5
COST_MOVE = 1.0
BOUNDRY_PENALTY = 0.1
COST_DIAG_MOVE = 1.5
ACTION_ARRAY = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]
ACTION_COST = [COST_NOOP, COST_MOVE, COST_MOVE, COST_MOVE, COST_MOVE, COST_DIAG_MOVE, COST_DIAG_MOVE, COST_DIAG_MOVE, COST_DIAG_MOVE]
SAFE_REWARDS = 2.0


class TestSimpleMOEnv(gym.Env, EzPickle):
    """
    A MO-enviroment with 3 stationary goals, an exit, the agent has to collect at least one goal before exiting the stage
    Rewards are: Speed to Exit, Goals Completed
    """

    metadata = {"render_modes": ["human", "rgb_array"], "video.frames_per_second": 3, "render_fps": 1}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, render_mode="human", width=2, height=2, fps=3, max_steps=200):
        self.metadata["video.frames_per_second"] = 1
        self.render_mode = render_mode
        self.frames_per_second = fps
        self.viewer = None
        self.width = min(width, 2)
        self.height = min(height, 2)
        self.max_steps = max_steps

        self.steps = 0
        self.agent = (self.width - 1, self.height - 1)

        self.reward_space_info = ["water", "food"]
        self.reward_space = Box(low=np.array([-2.0, -2.0]), high=np.array([2.0, 2.0]), shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0, high=max(self.width, self.height), shape=(2,), dtype=np.int32)
        self.action_space = Discrete(9)  # noop, up, down, left, right, up-left, up-right, down-left, down-right = 0, 1, 2, 3, 4, 5, 6, 7

    def step(self, action):

        action_move = ACTION_ARRAY[action]
        reward = [-ACTION_COST[action], -ACTION_COST[action]]
        self.steps += 1

        new_location = (self.agent[0] + action_move[0], self.agent[1] + action_move[1])
        done = False
        if new_location[0] >= 0 and new_location[0] < self.width and new_location[1] >= 0 and new_location[1] < self.height:
            self.agent = new_location
        else:
            reward[0], reward[1] = reward[0] - BOUNDRY_PENALTY, reward[1] - BOUNDRY_PENALTY

        if self.agent[0] == 0:
            reward[0] = reward[0] + SAFE_REWARDS

        if self.agent[1] == 0:
            reward[1] = reward[1] + SAFE_REWARDS

        return self.get_obs(), reward, done, self.steps >= self.max_steps, {}

    def get_obs(self):
        return np.array([self.agent[0], self.agent[1]], dtype=np.int32)

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.agent = (self.width - 1, self.height - 1)

        return self.get_obs(), {}

    def get_image(self):
        matrix = [[(255, 255, 255) for _ in range(self.width)] for _ in range(self.height)]
        for row in matrix:
            row[0] = (0, 180, 0)
        matrix[0] = [(0, 0, 180)] * len(matrix[0])
        matrix[0][0] = (0, 150, 150)
        matrix[self.agent[1]][self.agent[0]] = (0, 0, 0)
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

    def _is_valid_state(self, state):
        return state[0] >= 0 and state[0] < self.width and state[1] >= 0 and state[1] < self.height

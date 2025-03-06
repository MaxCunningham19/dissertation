import warnings

from matplotlib import pyplot as plt

from exploration_strategy import Greedy
from utils.plotting import plot_agent_actions_2d_seperated

warnings.filterwarnings("ignore")

import gym
from gym.wrappers.record_video import RecordVideo
import mo_gymnasium as mo_gym
import numpy as np
import torch
import pandas as pd

import envs  # This imports all the environments
from utils.constants import MODELS_DIR, RESULTS_DIR
from exploration_strategy.utils import create_exploration_strategy
from utils import extract_kwargs, build_parser, run_env, plot_agent_actions_2d, plot_over_time_multiple_subplots, smooth, kwargs_to_string
from agents import get_agent, DWL
from utils import generate_file_structure, kwargs_to_string
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_load_model", type=str, default=None, help="the path to load the model from")
parser.add_argument("--plot", type=bool, default=False, help="whether to plot the results")
args = parser.parse_args()

env_name = f"mo-deep-sea-treasure-concave-v0"

env = mo_gym.make(env_name, render_mode="rgb_array")
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = env.unwrapped.reward_space.shape[0]

deep_sea_treasure_labels = ["treasure value", "time penalty"]


# Setup agent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = DWL(input_shape=env.observation_space.shape, num_actions=n_action, num_policies=n_policy, exploration_strategy=Greedy(), device=device)
agent.load(args.path_to_load_model)


low = env.observation_space.low.astype(np.int32)
high = env.observation_space.high.astype(np.int32)
rows = high[1] - low[1]
cols = high[0] - low[0]
states = [[0.0] * cols for _ in range(rows)]

colors = plt.cm.viridis(np.linspace(0, 1, n_policy + 1))
xs = np.arange(n_action)
# Handle single subplot case

fig, axes = plt.subplots(len(states), len(states[0]))
if not hasattr(axes, "__len__"):
    axes = [axes]
else:
    axes = axes.flatten()
bar_width_single = 0.1 / n_policy
if bar_width_single < 0.1:
    bar_width_single = 0.1
negative_offset = -bar_width_single * ((n_policy) // 2)
legend = {}
legend_set = False

objective_labels = [f"objective {i+1}" for i in range(n_policy)]
for y, row in enumerate(states):
    for x, value in enumerate(row):
        # Get the correct axis based on whether we have a single subplot or multiple
        current_ax = axes[y * len(states[0]) + x]
        current_ax.set_xticks([])
        current_ax.set_yticks([])
        if x == 0:
            current_ax.set_ylabel(f"{y}")
        if y == len(states) - 1:
            current_ax.set_xlabel(f"{x}")
        idx = np.array([x, y])
        if env.unwrapped._is_valid_state((idx[1], idx[0])):
            weights, q_valuess = agent.get_all_info(idx)
            for i, q_values in enumerate(q_valuess):
                offset = negative_offset + (i) * (bar_width_single)
                bar = current_ax.bar(xs + offset, q_values, width=bar_width_single, label=objective_labels[i], color=colors[i], alpha=0.7)
                legend[objective_labels[i]] = bar
                for rect in bar:
                    height = rect.get_height()
                    current_ax.text(rect.get_x() + rect.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=2)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.13, wspace=0.01, hspace=0.01)
plt.figlegend(legend.values(), legend.keys(), loc="lower center", ncol=n_policy, bbox_to_anchor=(0.5, 0.02), fontsize=8)
plt.savefig(f"./objective_q_values.png", dpi=500)
if args.plot:
    plt.show()
plt.close()

colors = plt.cm.viridis(np.linspace(0, 1, n_policy))
legend = {}
fig, axes = plt.subplots(len(states), len(states[0]))
if not hasattr(axes, "__len__"):
    axes = [axes]
else:
    axes = axes.flatten()
bar_width = 0.2
for y, row in enumerate(states):
    for x, value in enumerate(row):
        current_ax = axes[y * len(states[0]) + x]
        current_ax.set_xticks([])
        current_ax.set_yticks([])
        if x == 0:
            current_ax.set_ylabel(f"{y}")
        if y == len(states) - 1:
            current_ax.set_xlabel(f"{x}")
        idx = np.array([x, y])
        if env.unwrapped._is_valid_state((idx[1], idx[0])):
            weights, q_values = agent.get_all_info(idx)
            # Set background color based on max weight
            max_weight_idx = np.argmax(weights)
            current_ax.set_facecolor(colors[max_weight_idx])

            # Add bar chart of weights
            xs = np.arange(len(weights))
            bars = current_ax.bar(xs, weights, width=bar_width, color="grey", alpha=0.7)
            color = "white" if max_weight_idx > len(colors) / 2 else "black"
            for i, rect in enumerate(bars):
                legend[objective_labels[i]] = rect
                height = rect.get_height()
                current_ax.text(rect.get_x() + rect.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=3, color=color)
            # Add text showing the weight value
            current_ax.set_title(f"{weights[max_weight_idx]:.2f}", ha="center", va="top", fontsize=4, pad=1)

for i in range(len(colors)):
    legend[f"objective {i+1}"] = plt.Rectangle((0, 0), 1, 1, color=colors[i])

plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.13, wspace=0.25, hspace=0.25)
plt.figlegend(legend.values(), legend.keys(), loc="lower center", ncol=n_policy, bbox_to_anchor=(0.5, 0.02), fontsize=8)
plt.savefig(f"./objective_weights.png", dpi=500)
if args.plot:
    plt.show()

plt.close()
env.close()

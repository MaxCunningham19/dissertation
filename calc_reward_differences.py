import os
import torch
import warnings

warnings.filterwarnings("ignore")

from agents.utils import get_agent
from exploration_strategy.utils import create_exploration_strategy

import argparse
import gym
import mo_gymnasium as mo_gym
import numpy as np
import envs
from utils.utils import extract_kwargs
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True, help="the environment to run")
parser.add_argument("--model", type=str, required=True, help="MORL model to use: dwn, democratic, dueling")
parser.add_argument("--model_path", type=str, required=True, help="path to the model to plot")
parser.add_argument(
    "--model_kwargs",
    type=str,
    nargs="*",
    required=False,
    help="model specific parameters not provided below e.g. --model_kwargs arg1=value1 arg2=value2",
)
parser.add_argument("--objective_labels", type=str, nargs="*", default=None, help="objective labels")
parser.add_argument("--action_labels", type=str, nargs="*", default=None, help="action labels")
parser.add_argument("--images_dir", type=str, default="images", help="images directory")
parser.add_argument("--plot", action="store_true", default=False, help="plot the results")
parser.add_argument("--max_steps", type=int, required=False, default=100, help="the number of steps to run an episode for")
args = parser.parse_args()


env = mo_gym.make(args.env, render_mode="rgb_array", max_episode_steps=args.max_steps)
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = env.unwrapped.reward_space.shape[0]

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
agent_name = " ".join(args.model.split(" ")).lower()
model_kwargs = extract_kwargs(args.model_kwargs)
model_kwargs["device"] = device
model_kwargs["input_shape"] = env.observation_space.shape
model_kwargs["num_actions"] = n_action
model_kwargs["num_policies"] = n_policy
model_kwargs["exploration_strategy"] = create_exploration_strategy("greedy")
model_kwargs["memory_size"] = 1
if "w_exploration_strategy" in model_kwargs:
    model_kwargs["w_exploration_strategy"] = create_exploration_strategy("greedy")
agent = get_agent(agent_name, **model_kwargs)

if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model file not found: {args.model_path}")
agent.load(args.model_path)

if not os.path.exists(args.images_dir):
    os.makedirs(args.images_dir)

import numpy as np

combinations = []
for i in np.arange(0.0, 1.1, 0.1):
    combinations.append(round(i, 1))

from itertools import product

perms = product(combinations, repeat=n_policy)
first = False
human_preferences = []
for perm in perms:
    perm = list(map(lambda x: round(x, 1), perm))
    if round(sum(perm), 1) == 1.0:
        human_preferences.append(list(perm))

if (100.0 // n_policy) % 10 != 0:
    human_preferences.append([round(1.0 / n_policy, 2)] * n_policy)

human_preferences = sorted(human_preferences)
results = []
for human_preference in human_preferences:
    episode_reward = [0.0] * n_policy
    done = truncated = False

    obs, _ = env.reset()
    while not (done or truncated):
        action, info = agent.get_action(obs, human_preference)
        obs_, reward, done, truncated, _ = env.step(action)
        obs = obs_
        for j in range(n_policy):
            episode_reward[j] = episode_reward[j] + reward[j]

    results.append(human_preference + episode_reward)

path = f"{args.images_dir}/{args.env}"
csv_file = f"{path}/episode_rewards.csv"
write_header = not os.path.exists(csv_file)
if not os.path.exists(path):
    os.makedirs(path)

with open(csv_file, mode="a", newline="") as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(
            ["Agent"] + [f"Human Preference Objective {i}" for i in range(n_policy)] + [f"Episode Reward Objective {i}" for i in range(n_policy)]
        )
    for result in results:
        writer.writerow([agent.name()] + result)

if args.plot:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_file)

    reward_cols = [col for col in df.columns if "Episode Reward Objective" in col]
    pref_cols = [col for col in df.columns if "Human Preference Objective" in col]
    n_policy = len(reward_cols)

    df["Preference Label"] = df[pref_cols].apply(lambda row: "(" + ", ".join(map(str, row)) + ")", axis=1)

    agents = df["Agent"].unique()
    x = np.arange(len(df[df["Agent"] == agents[0]]))
    width = 0.25

    for i, reward_col in enumerate(reward_cols):
        fig, ax = plt.subplots(figsize=(14, 7))
        for j, agent in enumerate(agents):
            agent_data = df[df["Agent"] == agent]
            offset = (j - (len(agents) - 1) / 2) * width  # Center bars
            ax.bar(x + offset, agent_data[reward_col], width, label=agent)

        ax.set_title(f"Episode Rewards for Objective {i}")
        ax.set_xlabel("Human Preference Vector")
        ax.set_ylabel("Reward")
        ax.grid(False)
        ax.set_xticks(x)
        ax.set_xticklabels(df[df["Agent"] == agents[0]]["Preference Label"], rotation=45, ha="right")
        ax.legend()
        ax.grid(True, axis="y")
        plt.tight_layout()

        file_path = f"{path}/reward_objective_{i}.png"
        plt.savefig(file_path)
        print(f"Saved plot to {file_path}")
        plt.close()

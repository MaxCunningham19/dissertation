import os
import torch
import warnings

warnings.filterwarnings("ignore")

from agents.utils import get_agent
from exploration_strategy.utils import create_exploration_strategy
from utils.plotting import plot_agent_actions, plot_agent_objective_q_values, plot_agent_objective_q_values_seperated, plot_agent_q_values

import argparse
import gym
from gym.wrappers.record_video import RecordVideo
import mo_gymnasium as mo_gym
import numpy as np
import envs
from utils.utils import extract_kwargs

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
parser.add_argument("--human_preference", type=float, nargs="*", default=None, required=False, help="an array of human preferences of objectives")
args = parser.parse_args()


env = mo_gym.make(args.env, render_mode="rgb_array")
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
if "w_exploration_strategy" in model_kwargs:
    model_kwargs["w_exploration_strategy"] = create_exploration_strategy("greedy")
agent = get_agent(agent_name, **model_kwargs)

if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model file not found: {args.model_path}")
agent.load(args.model_path)

if not os.path.exists(args.images_dir):
    os.makedirs(args.images_dir)

low = env.observation_space.low.astype(np.int32)
high = env.observation_space.high.astype(np.int32)
rows = high[1] - low[1]
cols = high[0] - low[0]
states = []
for row in range(rows):
    state_row = []
    for col in range(cols):
        state_row.append(np.array([col, row]))
    states.append(state_row)
if "deep-sea-treasure" in args.env:
    states = []
    for row in range(rows):
        state_row = []
        for col in range(cols):
            state_row.append(np.array([row, col]))
        states.append(state_row)

plot_agent_objective_q_values(
    states,
    agent,
    n_action,
    n_policy,
    save_path=f"{args.images_dir}",
    plot=args.plot,
    objective_labels=args.objective_labels,
    should_plot=lambda state: env.unwrapped._is_valid_state(state),
    human_preference=args.human_preference,
)

plot_agent_objective_q_values_seperated(
    states,
    agent,
    n_action,
    n_policy,
    save_path=f"{args.images_dir}",
    objective_labels=args.objective_labels,
    plot=args.plot,
    should_plot=lambda state: env.unwrapped._is_valid_state(state),
    human_preference=args.human_preference,
)

plot_agent_q_values(
    states,
    agent,
    n_action,
    n_policy,
    save_path=f"{args.images_dir}",
    plot=args.plot,
    objective_labels=args.objective_labels,
    should_plot=lambda state: env.unwrapped._is_valid_state(state),
    human_preference=args.human_preference,
)

plot_agent_actions(
    states,
    agent,
    n_action,
    n_policy,
    save_path=f"{args.images_dir}",
    plot=args.plot,
    objective_labels=args.objective_labels,
    action_labels=args.action_labels,
    should_plot=lambda state: env.unwrapped._is_valid_state(state),
    human_preference=args.human_preference,
)

env.close()

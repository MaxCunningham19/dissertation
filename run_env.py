import gym
from gym.wrappers.record_video import RecordVideo
import mo_gymnasium as mo_gym
from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from action_selection import EpsilonGreedy
from action_selection.utils import create_exploration_strategy
from utils import extract_kwargs, softmax, build_parser, run_env
from agents.democratic.democratic_dqn_4ly import DemocraticDQN
from agents.dwn.DWL_4ly import DWL
from utils.plotting import plot_agent_actions_2d, plot_over_time_multiple_subplots, smooth

parser = build_parser()
args = parser.parse_args()

# Setup environment
num_episodes = args.num_episodes
num_episodes_to_record = args.num_record
interval = num_episodes // num_episodes_to_record
if interval == 0:
    interval = 1

env = mo_gym.make(args.env, render_mode="rgb_array", **extract_kwargs(args.env_kwargs))
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = env.unwrapped.reward_space.shape[0]

if args.record:
    env = RecordVideo(env, "videos/demo", episode_trigger=lambda e: e % interval == 0, name_prefix=args.env)

# Setup agent
agent_name = "".join(args.model.split(" ")).lower()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent_dictionary = {"democratic": DemocraticDQN}

if agent_name not in agent_dictionary:
    print("Invalid model selection")
    exit(1)

exploration_strategy = create_exploration_strategy(args.exploration, **extract_kwargs(args.exploration_kwargs))

agent = agent_dictionary[agent_name](
    env.observation_space.shape, n_action, n_policy, exploration_strategy, device=device, **extract_kwargs(args.model_kwargs)
)

if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)


# Run environment
episode_rewards, loss, csv_data = run_env(num_episodes, env, agent, n_policy)

agent.save(f"{args.path_to_save_model}/{args.model}")

headers = ["episode", "episode_reward", "loss"]
df = pd.DataFrame(csv_data, columns=headers)
df.to_csv(args.path_to_csv_save, index=False)

# performance measurements
if args.plot:

    loss = np.array(loss).T
    plot_over_time_multiple_subplots(n_policy, loss)

    episode_rewards = np.array(episode_rewards).T

    window_size = 50
    for i, _ in enumerate(episode_rewards):
        episode_rewards[i] = smooth(episode_rewards[i])

    plot_over_time_multiple_subplots(n_policy, episode_rewards)

    low = env.observation_space.low.astype(np.int32)
    high = env.observation_space.high.astype(np.int32)
    rows = high[1] - low[1]
    cols = high[0] - low[0]
    states = [[0.0] * cols for _ in range(rows)]
    plot_agent_actions_2d(states, agent, n_action)


env.close()

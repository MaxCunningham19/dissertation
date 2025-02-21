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
from utils import extract_kwargs, softmax
from agents.democratic.democratic_dqn_4ly import DemocraticDQN
from agents.dwn.DWL_4ly import DWL

parser = argparse.ArgumentParser()
parser.add_argument("--num_episodes", type=int, default=1000, help="number of episodes")
parser.add_argument("--plot", action="store_true", default=False, help="Enable plotting the metrics calculated")
parser.add_argument("--record", action="store_true", default=False, help="Enable recording of episodes")
parser.add_argument("--num_record", type=int, default=25, help="number of episodes to record")

parser.add_argument("--path_to_load_model", type=str, help="path to load a network from")
parser.add_argument("--path_to_save_model", type=str, default="./agents/savedNets/", help="path to save the network to")
parser.add_argument("--path_to_csv_save", type=str, default="./output.csv", help="path to save the csv results to")

parser.add_argument("--model", type=str, default="democratic", help="MORL model to use: dwn, democratic, dueling")
parser.add_argument(
    "--model_kwargs", type=str, nargs="*", help="model specific parameters not provided below e.g. --model_kwargs arg1=value1 arg2=value2"
)

parser.add_argument("--exploration", type=str, default="epsilon", help="exploration strategy for deciding which action to take")
parser.add_argument(
    "--exploration_kwargs",
    type=str,
    nargs="*",
    help="exploration strategy specific parameters not provided below e.g. --exploration_kwargs arg1=value1 arg2=value2",
)

parser.add_argument("--env", type=str, default="deep-sea-treasure-v0", help="the mo-gymnasium environment to train the agent on")
parser.add_argument("--env_kwargs", type=str, nargs="*", help="the key value pair arguments for the environment e.g. key1=value1")

parser.add_argument(
    "--not_training",
    action="store_true",
    default=False,
    help="set this flag if you do not want the agent to train, used to evaluate a trained agents execution",
)

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
csv_data = []
loss = []
episode_rewards = []
for i in range(num_episodes + 1):  # + 1 too ensure the final episode is saved
    episode_reward = np.array([0.0] * n_policy)
    done = truncated = False

    obs, info = env.reset()
    while not (done or truncated):
        action = agent.get_action(obs)

        obs_, reward, done, truncated, info = env.step(action)
        agent.store_memory(obs, action, reward, obs_, done)
        obs = obs_

        episode_reward = episode_reward + np.array(reward)

    episode_rewards.append(episode_reward)
    loss_info = agent.get_loss_values()
    loss.append(loss_info)
    csv_data.append([i, episode_reward, loss_info])

    if i % 10 == 0:
        print("Epsiode", i, episode_reward, loss_info)
    agent.train()

agent.save(f"{args.path_to_save_model}/{args.model}")

headers = ["episode", "episode_reward", "loss"]
df = pd.DataFrame(csv_data, columns=headers)
df.to_csv(args.path_to_csv_save, index=False)

# performance measurements
if args.plot:
    plt.plot(loss)
    plt.xlabel("episodes")
    plt.ylabel("loss")
    plt.show()

    episode_rewards = np.array(episode_rewards).T

    window_size = 50

    fig, axes = plt.subplots(n_policy, 1)
    for i, col in enumerate(episode_rewards):
        smoothed_rewards = np.convolve(col, np.ones(window_size) / window_size, mode="valid")
        x_values = np.arange(len(smoothed_rewards)) + window_size // 2
        # axes[i].plot(col, label="Raw Rewards", alpha=0.3, color="blue")
        axes[i].plot(x_values, smoothed_rewards, label="Moving Avg (50 eps)", color="red")
        axes[i].set_xlabel("episodes")
        axes[i].set_ylabel(f"reward")
        axes[i].set_title(f"{i}")

    plt.tight_layout()
    plt.show()

    xs = np.arange(n_action)
    bar_width = 0.2

    obs_space = env.observation_space

    low = obs_space.low.astype(np.int32)
    high = obs_space.high.astype(np.int32)

    rows = high[1] - low[1]
    cols = high[0] - low[0]
    fig, axes = plt.subplots(len(CONCAVE_MAP), len(CONCAVE_MAP[0]))
    CONCAVE_MAP
    print(CONCAVE_MAP)
    for x in range(rows):
        for y in range(cols):
            ax = axes[x, y]
            ax.set_xticks([])
            ax.set_yticks([])
            # if value >= 0.0 or value >= 0:
            idx = np.array([x, y])
            agent_info = agent.get_agent_info(idx)
            advantages1 = agent_info[0]
            advantages2 = agent_info[1]
            advantages1 = advantages1.tolist()[0]
            advantages2 = advantages2.tolist()[0]
            soft_array1 = softmax(advantages1)
            soft_array2 = softmax(advantages2)
            array1 = advantages1 - np.mean(advantages1)
            array2 = advantages2 - np.mean(advantages2)
            # Plot bars in the current subplot
            ax.bar(xs - bar_width / 2, soft_array1, width=bar_width, label="treasure", color="blue", alpha=0.7)
            ax.bar(xs + bar_width / 2, soft_array2, width=bar_width, label="speed", color="green", alpha=0.7)

    plt.legend()
    plt.show()


env.close()

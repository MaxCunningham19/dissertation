import gym
from gym.wrappers.record_video import RecordVideo
import mo_gymnasium as mo_gym
from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from agents.democratic.democratic_dqn_4ly import DemocraticDQN


parser = argparse.ArgumentParser()
parser.add_argument("--num_episodes", type=int, default=1000, help="number of episodes")
parser.add_argument("--num_episodes_to_record", type=int, default=25, help="number of episodes to record")
parser.add_argument("--max_steps", type=int, default=25, help="maximum number of steps per episode")
parser.add_argument("--plot", action="store_true", default=False, help="Enable plotting the metrics calculated")
parser.add_argument("--path_to_load_model", type=str, help="path to load a network from")
parser.add_argument("--path_to_save_model", type=str, default="./agents/savedNets/demo/test_env", help="path to save the network to")
parser.add_argument("--path_to_csv_save", type=str, default="./output.csv", help="path to save the csv results to")


args = parser.parse_args()

# Setup environment
num_episodes = args.num_episodes
num_episodes_to_record = args.num_episodes_to_record
interval = num_episodes // num_episodes_to_record
if interval == 0:
    interval = 1

environment_name = "deep-sea-treasure-v0"
reward_space_info = ["treasure", "time_penalty"]
env = mo_gym.make("deep-sea-treasure-v0", render_mode="rgb_array", dst_map=CONCAVE_MAP)
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = 2

env = RecordVideo(env, "videos/demo", episode_trigger=lambda e: e % interval == 0, name_prefix=environment_name)
# Setup agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = DemocraticDQN(
    env.observation_space.shape,
    n_action,
    n_policy,
    device=device,
    batch_size=1048,
    learning_rate=0.001,
    gamma=0.99,
    alpha=1.0,
    hidlyr_nodes=258,
    human_preference=[0.5, 0.5],
    tau=0.001,
)

if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)

csv_data = []
loss = []
episode_rewards = []
for i in range(num_episodes + 1):
    episode_reward = np.array([0.0] * n_policy)
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        actions = agent.get_actions(obs)
        action = agent.get_action(obs, False, False)
        obs_, reward, done, truncated, info = env.step(action)
        agent.store_memory(obs, action, reward, obs_, done)
        episode_reward = episode_reward + np.array(reward)
        obs = obs_

    episode_rewards.append(episode_reward)
    loss_info = agent.get_loss_values()
    loss.append(loss_info)
    csv_data.append([i, episode_reward, loss_info])
    # agent.train()

print(episode_rewards)
agent.save(args.path_to_save_model)

xs = np.arange(n_action)
bar_width = 0.2


def softmax(x):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


obs_space = env.observation_space  # Get the Box space

low = obs_space.low.astype(np.int32)  # Convert to integer bounds
high = obs_space.high.astype(np.int32)  # Convert to integer bounds

rows = high[1] - low[1] + 1  # Height of grid (Y dimension)
cols = high[0] - low[0] + 1  # Width of grid (X dimension)
fig, axes = plt.subplots(len(CONCAVE_MAP), len(CONCAVE_MAP[0]))
CONCAVE_MAP
print(CONCAVE_MAP)
# Iterate through all integer states
for x, row in enumerate(CONCAVE_MAP):  # +1 to include the upper bound
    for y, value in enumerate(row):
        print(x, y, value)
        ax = axes[x, y]
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        if value >= 0.0 or value >= 0:
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

headers = ["episode", "episode_reward", "loss"]
df = pd.DataFrame(csv_data, columns=headers)
df.to_csv(args.path_to_csv_save, index=False)


if args.plot:
    plt.plot(loss)
    plt.xlabel("episodes")
    plt.ylabel("loss")
    plt.show()

    episode_rewards = np.array(episode_rewards).T

    window_size = 50

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    fig, axes = plt.subplots(n_policy, 1)
    for i, col in enumerate(episode_rewards):
        smoothed_rewards = moving_average(col, window_size)
        x_values = np.arange(len(smoothed_rewards)) + window_size // 2
        # axes[i].plot(col, label="Raw Rewards", alpha=0.3, color="blue")
        axes[i].plot(x_values, smoothed_rewards, label="Moving Avg (50 eps)", color="red")
        axes[i].set_xlabel("episodes")
        axes[i].set_ylabel(f"reward")
        axes[i].set_title(f"{reward_space_info[i]}")
        # axes[i].legend()

    plt.tight_layout()
    plt.show()


env.close()

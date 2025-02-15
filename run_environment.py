import gym
from gym.wrappers.record_video import RecordVideo
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import argparse


from agents.dueling_dqn_agent_4ly import DUELING_DQN
from agents.dueling_dwn_4ly import DWL
import test_env

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
env = gym.make("simplemoenv", render_mode="rgb_array", max_steps=args.max_steps)
env = RecordVideo(env, "videos/demo", episode_trigger=lambda e: e % interval == 0, name_prefix="video-cartpole")

# Setup agent
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = env.reward_space.shape[0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = DWL(env.observation_space.shape, n_action, n_policy, device=device, batch_size=64, learning_rate=0.1)

if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)

csv_data = []
loss = []
episode_rewards = []
for i in range(num_episodes + 1):
    if i % (100) == 0:
        print("Episode: ", i)
    episode_reward = np.array([0.0] * n_policy)
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action = agent.get_action(obs, False)
        obs_, reward, done, truncated, info = env.step(action)
        if i == num_episodes:
            print(obs, action, reward, obs_)
        agent.store_memory(obs, action, reward, obs_, done)
        episode_reward = episode_reward + np.array(reward)
        obs = obs_
        agent.train()

    episode_rewards.append(episode_reward)
    loss_info = agent.get_loss_values()
    loss.append(loss_info)
    csv_data.append([i, episode_reward, loss_info])

# print(episode_rewards)
agent.save(args.path_to_save_model)

labels = ["noop", "up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right"]
x = np.arange(len(labels))
bar_width = 0.2
fig, axes = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        ax = axes[i, j]
        idx = np.array([i, j])
        agent_info = agent.get_agent_info(idx)
        _, value1, advantages1 = agent_info[0]
        _, value2, advantages2 = agent_info[1]
        advantages1 = advantages1.tolist()[0]
        advantages2 = advantages2.tolist()[0]
        value1 = value1.tolist()[0]
        value2 = value2.tolist()[0]
        array1 = advantages1 - np.mean(advantages1)
        array2 = advantages2 - np.mean(advantages2)
        # Plot bars in the current subplot
        ax.bar(x - bar_width / 2, array1, width=bar_width, label="Agent width", color="blue", alpha=0.7)
        ax.bar(x + bar_width / 2, array2, width=bar_width, label="Agent height", color="green", alpha=0.7)

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_title(f"Subplot ({i}, {j})")
        ax.set_ylabel("Value")

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

    fig, axes = plt.subplots(n_policy, 1)
    for i, col in enumerate(episode_rewards):
        axes[i].plot(col)
        axes[i].set_xlabel("episodes")
        axes[i].set_ylabel(f"reward")
        axes[i].set_title(f"{env.reward_space_info[i]}")

    plt.tight_layout()
    plt.show()


env.close()

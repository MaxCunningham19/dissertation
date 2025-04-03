import os
import warnings

warnings.filterwarnings("ignore")

import gym
from gym.wrappers.record_video import RecordVideo
import mo_gymnasium as mo_gym
import numpy as np
import torch
import pandas as pd

import envs  # This imports all the environments
from exploration_strategy.utils import create_exploration_strategy
from utils import extract_kwargs, build_parser, plot_over_time_multiple_subplots, smooth, kwargs_to_string
from agents import get_agent
from utils import generate_file_structure, kwargs_to_string

parser = build_parser()
parser.add_argument("--env", type=str, required=True, help="the environment to run")
args = parser.parse_args()


# Setup environment
num_episodes = args.num_episodes
num_episodes_to_record = args.num_record
interval = num_episodes // num_episodes_to_record
if interval == 0:
    interval = 1


env = mo_gym.make(args.env, render_mode="rgb_array", max_episode_steps=args.max_steps)
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = env.unwrapped.reward_space.shape[0]

# Setup agent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exploration_strategy = create_exploration_strategy(args.exploration, **extract_kwargs(args.exploration_kwargs))
model_kwargs = extract_kwargs(args.model_kwargs)
model_kwargs["device"] = device
model_kwargs["input_shape"] = env.observation_space.shape
model_kwargs["num_actions"] = n_action
model_kwargs["num_policies"] = n_policy
model_kwargs["exploration_strategy"] = exploration_strategy
agent_name = " ".join(args.model.split(" ")).lower()
print(args.model, agent_name)
agent = get_agent(agent_name, **model_kwargs)


if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)


results_dir, images_dir, models_dir, videos_dir = generate_file_structure(
    args.env, "", args.model, kwargs_to_string(model_kwargs), args.exploration, kwargs_to_string(args.exploration_kwargs)
)
results_path = f"{results_dir}/results.csv"
start_episode = 0

df = pd.DataFrame(columns=["episode", "episode_reward", "loss"])
if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        start_episode = df.iloc[-1]["episode"] + 1

if args.record:
    env = RecordVideo(env, videos_dir, episode_trigger=lambda e: e % interval == 0)
try:

    for i in range(start_episode, num_episodes):
        episode_reward = [0.0] * n_policy
        done = truncated = False

        obs, _ = env.reset()
        while not (done or truncated):
            action, info = agent.get_action(obs)
            obs_, reward, done, truncated, _ = env.step(action)
            agent.store_memory(obs, action, reward, obs_, done, info)
            obs = obs_
            for j in range(n_policy):
                episode_reward[j] = episode_reward[j] + reward[j]

        loss_info = agent.get_loss_values()
        # Append new row to DataFrame
        df.loc[len(df)] = [i, episode_reward, loss_info]

        if i % 10 == 0:
            print(f"Episode {i} completed")

        agent.train()
        agent.update_params()
    print(f"Episode {num_episodes} completed")
except (Exception, KeyboardInterrupt) as e:
    if isinstance(e, KeyboardInterrupt):
        print("\nTraining interrupted by user")
    else:
        print(f"An error occurred during training: {str(e)}\n {e.with_traceback()}")
    response = input("Do you want to save the current results and abort? (Y/n): ")
    if response.lower() == "n":
        print("Aborting...")
        exit(1)
    else:
        print("Saving results and finishing...")
        pass

agent.save(f"{models_dir}")

# Save DataFrame to CSV
df.to_csv(results_path, index=False)


# Before plotting, convert the string representations of lists to actual numpy arrays
def parse_list_string(s):
    try:
        # Remove brackets and split by comma
        s = s.strip("[]").split(",")
        # Convert to float array
        return np.array([float(x) for x in s])
    except:
        return np.array([np.nan, np.nan])  # Return nan array if parsing fails


df = pd.read_csv(results_path)
loss_arrays = df["loss"].apply(parse_list_string).values
loss = np.array([x for x in loss_arrays]).T

plot_over_time_multiple_subplots(n_policy, loss, save_path=f"{images_dir}/loss.png", plot=args.plot)

episode_rewards_arrays = df["episode_reward"].apply(parse_list_string).values
episode_rewards = np.array([x for x in episode_rewards_arrays]).T

plot_over_time_multiple_subplots(n_policy, episode_rewards, save_path=f"{images_dir}/rewards.png", plot=args.plot)

window_size = 50
smoothed_rewards = []
for i in range(n_policy):
    smoothed_rewards.append(smooth(episode_rewards[i], window_size))
plot_over_time_multiple_subplots(n_policy, smoothed_rewards, save_path=f"{images_dir}/smoother_rewards.png", plot=args.plot)


env.close()

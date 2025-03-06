import warnings

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
from agents import get_agent
from utils import generate_file_structure, kwargs_to_string

parser = build_parser()
parser.add_argument("--mountain_car_type", type=str, default="", help="the type of mountain car to use: 3d, timemove, timespeed")
parser.add_argument("--max_steps", type=int, default=200, help="the number of steps to run")
args = parser.parse_args()


# Ensure model is valid
agent_name = "".join(args.model.split(" ")).lower()
try:
    agent = get_agent(agent_name)
except ValueError as e:
    print(f"Invalid model selection: {e}")
    exit(1)

# Ensure exploration strategy is valid
exploration_strategy = None
try:
    exploration_strategy = create_exploration_strategy(args.exploration, **extract_kwargs(args.exploration_kwargs))
except Exception as e:
    print(f"Invalid exploration strategy: {e}")
    exit(1)


# Setup environment
num_episodes = args.num_episodes
num_episodes_to_record = args.num_record
interval = num_episodes // num_episodes_to_record
if interval == 0:
    interval = 1

env_name = f"mo-mountaincar-v0"
match args.mountain_car_type:
    case "3d":
        env_name = f"mo-mountaincar-3d-v0"
    case "timemove":
        env_name = f"mo-mountaincar-timemove-v0"
    case "timespeed":
        env_name = f"mo-mountaincar-timespeed-v0"

env = mo_gym.make(env_name, render_mode="rgb_array", max_episode_steps=args.max_steps)
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = env.unwrapped.reward_space.shape[0]

mountain_car_labels = ["time penalty", "reverse penalty", "forward penalty"]


results_dir, images_dir, models_dir, videos_dir = generate_file_structure(
    env_name, "", args.model, kwargs_to_string(args.model_kwargs), args.exploration, kwargs_to_string(args.exploration_kwargs)
)

if args.record:
    env = RecordVideo(env, videos_dir, episode_trigger=lambda e: e % interval == 0)

# Setup agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


exploration_strategy = create_exploration_strategy(args.exploration, **extract_kwargs(args.exploration_kwargs))

agent = agent(env.observation_space.shape, n_action, n_policy, exploration_strategy, device=device, **extract_kwargs(args.model_kwargs))

if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)

# Run environment
csv_data = []
loss = []
episode_rewards = []
state_counts = {}
action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
for i in range(num_episodes + 1):
    episode_reward = np.array([0.0] * n_policy)
    done = truncated = False
    obs, _ = env.reset()
    while not (done or truncated):
        action, info = agent.get_action(obs)
        obs_, reward, done, truncated, _ = env.step(action)
        action_counts[action] += 1
        state_counts[tuple(obs)] = state_counts.get(tuple(obs), 0) + 1
        agent.store_memory(obs, action, reward, obs_, done, info)
        obs = obs_
        episode_reward = episode_reward + np.array(reward)
    episode_rewards.append(episode_reward)
    loss_info = agent.get_loss_values()
    loss.append(loss_info)
    csv_data.append([i, episode_reward, loss_info])
    if i % 10 == 0:
        print("Epsiode", i, episode_reward, loss_info, sum(episode_rewards[len(episode_rewards) - 50 :]) / 50)
    agent.train()
    agent.update_params()


agent.save(f"{models_dir}")

headers = ["episode", "episode_reward", "loss"]
df = pd.DataFrame(csv_data, columns=headers)
df.to_csv(f"{results_dir}/results.csv", index=False)

# performance measurements
loss = np.array(loss).T
plot_over_time_multiple_subplots(n_policy, loss, save_path=f"{images_dir}/loss.png", plot=args.plot)

episode_rewards = np.array(episode_rewards).T
window_size = 50
for i, _ in enumerate(episode_rewards):
    episode_rewards[i] = smooth(episode_rewards[i])
plot_over_time_multiple_subplots(n_policy, episode_rewards, save_path=f"{images_dir}/rewards.png", plot=args.plot)


env.close()

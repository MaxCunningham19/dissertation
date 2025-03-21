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
parser.add_argument("--env", type=str, required=True, help="the environment to use: mountain_car, deep_sea_treasure")
parser.add_argument(
    "--env_kwargs", type=str, nargs="*", help="environment specific parameters not provided below e.g. --env_kwargs arg1=value1 arg2=value2"
)
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

env = mo_gym.make(args.env, render_mode="rgb_array", **extract_kwargs(args.env_kwargs))
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = env.unwrapped.reward_space.shape[0]

deep_sea_treasure_labels = ["time penalty", "treasure value", "fuel"]
objective_labels = None
if "deep-sea-treasure" in args.env:
    objective_labels = deep_sea_treasure_labels

results_dir, images_dir, models_dir, videos_dir = generate_file_structure(
    args.env,
    kwargs_to_string(args.env_kwargs),
    args.model,
    kwargs_to_string(args.model_kwargs),
    args.exploration,
    kwargs_to_string(args.exploration_kwargs),
)

if args.record:
    env = RecordVideo(env, videos_dir, episode_trigger=lambda e: e % interval == 0, name_prefix=args.env)

# Setup agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


exploration_strategy = create_exploration_strategy(args.exploration, **extract_kwargs(args.exploration_kwargs))

agent = agent(env.observation_space.shape, n_action, n_policy, exploration_strategy, device=device, **extract_kwargs(args.model_kwargs))

if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)

# Run environment
episode_rewards, loss, csv_data = run_env(num_episodes, env, agent, n_policy)


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

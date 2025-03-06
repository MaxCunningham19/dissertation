import warnings

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
from agents import get_agent
from utils import generate_file_structure

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
from agents import get_agent, CDWL
from utils import generate_file_structure, kwargs_to_string

parser = build_parser()
args = parser.parse_args()

# Setup environment
num_episodes = args.num_episodes
num_episodes_to_record = args.num_record
interval = num_episodes // num_episodes_to_record
if interval == 0:
    interval = 1

env_name = f"custom-deep-sea-treasure"

env = mo_gym.make(env_name, render_mode="rgb_array")
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = env.unwrapped.reward_space.shape[0]

deep_sea_treasure_labels = ["treasure value", "time penalty"]


# Setup agent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = CDWL(env.observation_space.shape, n_action, n_policy, dnn_structure=False, **extract_kwargs(args.model_kwargs))


results_dir, images_dir, models_dir, videos_dir = generate_file_structure(
    env_name, "", args.model, kwargs_to_string(args.model_kwargs), args.exploration, kwargs_to_string(args.exploration_kwargs)
)

if args.record:
    env = RecordVideo(env, videos_dir, episode_trigger=lambda e: e % interval == 0)

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
plot_over_time_multiple_subplots(n_policy, episode_rewards, save_path=f"{images_dir}/true_rewards.png", plot=args.plot)
window_size = 50
smoothed_rewards = [None] * len(episode_rewards)
for i, episode_reward_objective in enumerate(episode_rewards):
    smoothed_rewards[i] = smooth(episode_reward_objective, window_size)
plot_over_time_multiple_subplots(n_policy, smoothed_rewards, save_path=f"{images_dir}/rewards.png", plot=args.plot)

env.close()
kwargs_to_string

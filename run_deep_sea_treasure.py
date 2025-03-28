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
from utils import generate_file_structure, kwargs_to_string

parser = build_parser()
parser.add_argument("--treasure_type", type=str, default="concave", help="the type of treasure to use: concave, convex, mirrored")
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

env_name = f"mo-deep-sea-treasure-{args.treasure_type}-v0"

env = mo_gym.make(env_name, render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = env.unwrapped.reward_space.shape[0]

deep_sea_treasure_labels = ["treasure value", "time penalty"]


# Setup agent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exploration_strategy = create_exploration_strategy(args.exploration, **extract_kwargs(args.exploration_kwargs))
agent = agent(
    env.observation_space.shape, n_action, n_policy, exploration_strategy=exploration_strategy, device=device, **extract_kwargs(args.model_kwargs)
)
if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)


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

low = env.observation_space.low.astype(np.int32)
high = env.observation_space.high.astype(np.int32)
rows = high[1] - low[1]
cols = high[0] - low[0]
states = [[0.0] * cols for _ in range(rows)]
plot_agent_actions_2d(
    states,
    agent,
    n_action,
    n_policy,
    save_path=f"{images_dir}/actions.png",
    plot=args.plot,
    objective_labels=deep_sea_treasure_labels,
    should_plot=lambda state: env.unwrapped._is_valid_state((state[1], state[0])),
)

plot_agent_actions_2d_seperated(
    states,
    agent,
    n_action,
    n_policy,
    save_path=f"{images_dir}",
    plot=args.plot,
    should_plot=lambda state: env.unwrapped._is_valid_state((state[1], state[0])),
)


env.close()

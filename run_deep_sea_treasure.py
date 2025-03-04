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
from agents import DemocraticDQN, DWL
from utils import generate_file_structure, kwargs_to_string

parser = build_parser()
parser.add_argument("--treasure_type", type=str, default="concave", help="the type of treasure to use: concave, convex, mirrored")
args = parser.parse_args()


agent_dictionary = {"democratic": DemocraticDQN, "dwl": DWL}

# Ensure model is valid
agent_name = "".join(args.model.split(" ")).lower()
if agent_name not in agent_dictionary:
    print(f"Invalid model selection: {args.model}")
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

env = mo_gym.make(env_name, render_mode="rgb_array")
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_policy = env.unwrapped.reward_space.shape[0]

deep_sea_treasure_labels = ["time penalty", "treasure value"]


results_dir, images_dir, models_dir, videos_dir = generate_file_structure(
    env_name, "", args.model, kwargs_to_string(args.model_kwargs), args.exploration, kwargs_to_string(args.exploration_kwargs)
)

if args.record:
    env = RecordVideo(env, videos_dir, episode_trigger=lambda e: e % interval == 0)

# Setup agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


exploration_strategy = create_exploration_strategy(args.exploration, **extract_kwargs(args.exploration_kwargs))

agent = agent_dictionary[agent_name](
    env.observation_space.shape, n_action, n_policy, exploration_strategy, device=device, **extract_kwargs(args.model_kwargs)
)

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

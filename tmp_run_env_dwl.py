import warnings

from utils.plotting import plot_agent_w_values

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
parser.add_argument("--w_exploration", type=str, default="epsilon", help="w_value exploration strategy for deciding which action to take")
parser.add_argument(
    "--w_exploration_kwargs",
    type=str,
    nargs="*",
    help="w_value exploration strategy specific parameters not provided below e.g. --w_exploration_kwargs arg1=value1 arg2=value2",
)
parser.add_argument("--w_check_interval", type=int, default=10, help="number of episodes to check w_values")
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
w_exploration_strategy = create_exploration_strategy(args.w_exploration, **extract_kwargs(args.w_exploration_kwargs))

model_kwargs = extract_kwargs(args.model_kwargs)
model_kwargs["device"] = device
model_kwargs["input_shape"] = env.observation_space.shape
model_kwargs["num_actions"] = n_action
model_kwargs["num_policies"] = n_policy
model_kwargs["exploration_strategy"] = exploration_strategy
model_kwargs["w_exploration_strategy"] = w_exploration_strategy

agent_name = args.model.join("").lower()
agent = get_agent(agent_name, **model_kwargs)


if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)


results_dir, images_dir, models_dir, videos_dir = generate_file_structure(
    args.env, "", args.model, kwargs_to_string(model_kwargs), args.exploration, kwargs_to_string(args.exploration_kwargs)
)

if args.record:
    env = RecordVideo(env, videos_dir, episode_trigger=lambda e: e % interval == 0)

low = env.observation_space.low.astype(np.int32)
high = env.observation_space.high.astype(np.int32)
rows = high[1] - low[1]
cols = high[0] - low[0]
states = []
for row in range(rows):
    state_row = []
    for col in range(cols):
        state_row.append(np.array([row, col]))
    states.append(state_row)

# Initialize w_values with more structure
w_values: dict[tuple[int, int], dict[int, list[np.float64]]] = {(i, j): {k: [] for k in range(n_policy)} for i in range(rows) for j in range(cols)}

csv_data = []
for i in range(num_episodes):
    episode_reward = np.array([0.0] * n_policy)
    done = truncated = False

    obs, _ = env.reset()
    while not (done or truncated):
        action, info = agent.get_action(obs)
        obs_, reward, done, truncated, _ = env.step(action)
        agent.store_memory(obs, action, reward, obs_, done, info)
        obs = obs_
        episode_reward = episode_reward + np.array(reward)

    loss_info = agent.get_loss_values()
    csv_data.append([i, episode_reward, loss_info])

    if i % 10 == 0:
        print(f"Episode {i} completed")

    if i % args.w_check_interval == 0:
        for row_idx in range(rows):
            for col_idx in range(cols):
                state = (row_idx, col_idx)
                if env.unwrapped._is_valid_state(state):
                    cur_s_w_values = agent.get_all_info(np.array([row_idx, col_idx]))[0]
                    for obj_idx, w_value in enumerate(cur_s_w_values):
                        w_values[state][obj_idx].append(np.float64(w_value))

    agent.train()
    agent.update_params()
print(f"Episode {i} completed")

agent.save(f"{models_dir}")

headers = ["episode", "episode_reward", "loss"]
df = pd.DataFrame(csv_data, columns=headers)
df.to_csv(f"{results_dir}/results.csv", index=False)

# performance measurements
loss = np.array(csv_data[:, 2]).T
plot_over_time_multiple_subplots(n_policy, loss, save_path=f"{images_dir}/loss.png", plot=args.plot)

episode_rewards = np.array(csv_data[:, 1]).T
plot_over_time_multiple_subplots(n_policy, episode_rewards, save_path=f"{images_dir}/rewards.png", plot=args.plot)

window_size = 50
smoothed_rewards = [None] * len(episode_rewards)
for i, episode_reward_objective in enumerate(episode_rewards):
    smoothed_rewards[i] = smooth(episode_reward_objective, window_size)
plot_over_time_multiple_subplots(n_policy, smoothed_rewards, save_path=f"{images_dir}/smoother_rewards.png", plot=args.plot)

valid_w_values = {state: values for state, values in w_values.items() if env.unwrapped._is_valid_state((state[0], state[1]))}

len_w_values_data = 0
for state, values in w_values.items():
    for obj_idx, valuez in values.items():
        len_w_values_data = max(len_w_values_data, len(valuez))

# Create empty array to store values
w_values_array = np.zeros((len(w_values) * len_w_values_data, n_policy))

row_idx = 0
for state, values in w_values.items():
    for t in range(len_w_values_data):
        for obj_idx in range(n_policy):
            if obj_idx in values and t < len(values[obj_idx]):
                w_values_array[row_idx, obj_idx] = values[obj_idx][t]
        row_idx += 1


# Create column names for objectives
column_names = [f"objective_{i}" for i in range(n_policy)]

w_values_df = pd.DataFrame(w_values_array, columns=column_names)
w_values_df.index.name = "timestep"

w_values_df.to_csv(f"{results_dir}/w_values.csv", index=False)
print(f"saved w_values to {results_dir}/w_values.csv")

plot_agent_w_values(states, valid_w_values, n_policy, save_path=f"{images_dir}", plot=args.plot)


env.close()

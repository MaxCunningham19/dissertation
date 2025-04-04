import os
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
parser.add_argument("--env_kwargs", type=str, nargs="*", help="environment kwargs")
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

env_kwargs = extract_kwargs(args.env_kwargs)
if "max_episode_steps" not in env_kwargs:
    env_kwargs["max_episode_steps"] = args.max_steps
env = mo_gym.make(args.env, render_mode="rgb_array", **env_kwargs)
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
agent_name = " ".join(args.model.split(" ")).lower()
print(args.model, agent_name)
agent = get_agent(agent_name, **model_kwargs)


if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)


results_dir, images_dir, models_dir, videos_dir = generate_file_structure(
    args.env, kwargs_to_string(env_kwargs), args.model, kwargs_to_string(model_kwargs), args.exploration, kwargs_to_string(args.exploration_kwargs)
)
results_path = f"{results_dir}/results.csv"
start_episode = 0

df = pd.DataFrame(columns=["episode", "episode_reward", "loss"])
if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    prev_results_path = "".join(args.path_to_load_model.split("models/") + ["results.csv"]).replace(" ", "")
    if os.path.exists(prev_results_path):
        df = pd.read_csv(prev_results_path)
        start_episode = df.iloc[-1]["episode"] + 1

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

w_values: dict[tuple[int, int], dict[int, list[np.float64]]] = {(i, j): {k: [] for k in range(n_policy)} for i in range(rows) for j in range(cols)}
if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    prev_w_values_path = "".join(args.path_to_load_model.split("models/") + ["w_values.csv"]).replace(" ", "")
    if os.path.exists(prev_w_values_path):
        w_values_df = pd.read_csv(prev_w_values_path)
        for _, row in w_values_df.iterrows():
            state = eval(row["state"])
            for obj_idx in range(n_policy):
                if f"objective_{obj_idx}" in row:
                    w_values[state][obj_idx].append(np.float64(row[f"objective_{obj_idx}"]))


if args.record:
    env = RecordVideo(env, videos_dir, episode_trigger=lambda e: (e + start_episode) % interval == 0, name_prefix=f"episode_{start_episode}")


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


valid_w_values = {state: values for state, values in w_values.items() if env.unwrapped._is_valid_state((state[0], state[1]))}

len_w_values_data = 0
for state, values in valid_w_values.items():
    for obj_idx, valuez in values.items():
        len_w_values_data = max(len_w_values_data, len(valuez))

# Create empty array to store values
w_values_array = []
row_idx = 0
for state, values in valid_w_values.items():
    for t in range(len_w_values_data):
        row = [state, t * args.w_check_interval]
        for obj_idx in range(n_policy):
            if obj_idx in values and t < len(values[obj_idx]):
                row.append(values[obj_idx][t])
            else:
                row.append(np.nan)
        w_values_array.append(row)
        row_idx += 1

print(w_values_array)
# Create column names for objectives
column_names = ["state", "timestep"] + [f"objective_{i}" for i in range(n_policy)]

w_values_df = pd.DataFrame(w_values_array, columns=column_names)
w_values_df.index.name = "timestep"

w_values_df.to_csv(f"{results_dir}/w_values.csv", index=False)

plot_agent_w_values(states, valid_w_values, n_policy, save_path=f"{images_dir}", plot=args.plot)


env.close()

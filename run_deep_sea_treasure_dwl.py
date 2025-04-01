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
from utils import (
    extract_kwargs,
    build_parser,
    plot_agent_actions_2d,
    plot_over_time_multiple_subplots,
    smooth,
    kwargs_to_string,
    plot_agent_w_values,
    plot_agent_actions_2d_seperated,
)
from agents import get_agent
from utils import generate_file_structure, kwargs_to_string

parser = build_parser()
parser.add_argument("--treasure_type", type=str, default="concave", help="the type of treasure to use: concave, convex, mirrored")
parser.add_argument("--w_exploration", type=str, default="greedy", help="the exploration strategy to use for the w values")
parser.add_argument("--w_exploration_kwargs", type=str, nargs="*", default=None, help="the kwargs for the w exploration strategy")
parser.add_argument("--w_value_rate", type=int, default=10, help="the rate at which to plot the w values")  # Increased frequency
args = parser.parse_args()


# Ensure model is valid
agent_name = "".join("dwl").lower()
try:
    agent = get_agent(agent_name)
except ValueError as e:
    print(f"Invalid model selection: {e}")
    exit(1)

# Ensure exploration strategy is valid
exploration_strategy = None
try:
    exploration_strategy = create_exploration_strategy(args.exploration, **extract_kwargs(args.exploration_kwargs))
    w_exploration_strategy = create_exploration_strategy(args.w_exploration, **extract_kwargs(args.w_exploration_kwargs))
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
    env.observation_space.shape,
    n_action,
    n_policy,
    exploration_strategy=exploration_strategy,
    w_exploration_strategy=w_exploration_strategy,
    device=device,
    **extract_kwargs(args.model_kwargs),
)
if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)


results_dir, images_dir, models_dir, videos_dir = generate_file_structure(
    env_name, "", args.model, kwargs_to_string(args.model_kwargs), args.exploration, kwargs_to_string(args.exploration_kwargs)
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
loss = []
episode_rewards = []
state_counts = {}
furthest_from_start = [0, 0]
for i in range(num_episodes + 1):
    episode_reward = np.array([0.0] * n_policy)
    done = truncated = False

    obs, _ = env.reset()
    while not (done or truncated):
        action, info = agent.get_action(obs)
        obs_, reward, done, truncated, _ = env.step(action)
        state_counts[tuple(obs)] = state_counts.get(tuple(obs), 0) + 1
        agent.store_memory(obs, action, reward, obs_, done, info)
        obs = obs_

        if obs[0] + obs[1] > furthest_from_start[0] + furthest_from_start[1]:
            furthest_from_start = obs

        episode_reward = episode_reward + np.array(reward)

    episode_rewards.append(episode_reward)
    loss_info = agent.get_loss_values()
    loss.append(loss_info)
    csv_data.append([i, episode_reward, loss_info])

    if i % 10 == 0:
        print("Epsiode", i, episode_reward, loss_info, sum(episode_rewards[len(episode_rewards) - 50 :]) / 50, furthest_from_start)
        furthest_from_start = [0, 0]

    # Record w-values more frequently and only for valid states
    if i % args.w_value_rate == 0:
        for row_idx in range(rows):
            for col_idx in range(cols):
                state = (row_idx, col_idx)
                if env.unwrapped._is_valid_state(state):
                    cur_s_w_values = agent.get_all_info(np.array([row_idx, col_idx]))[0]
                    for obj_idx, w_value in enumerate(cur_s_w_values):
                        w_values[state][obj_idx].append(np.float64(w_value))

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
plot_over_time_multiple_subplots(n_policy, episode_rewards, save_path=f"{images_dir}/true_rewards.png", plot=args.plot)
window_size = 50
smoothed_rewards = [None] * len(episode_rewards)
for i, episode_reward_objective in enumerate(episode_rewards):
    smoothed_rewards[i] = smooth(episode_reward_objective, window_size)
plot_over_time_multiple_subplots(n_policy, smoothed_rewards, save_path=f"{images_dir}/rewards.png", plot=args.plot)


plot_agent_actions_2d(
    states,
    agent,
    n_action,
    n_policy,
    save_path=f"{images_dir}/actions.png",
    plot=args.plot,
    objective_labels=deep_sea_treasure_labels,
    should_plot=lambda state: env.unwrapped._is_valid_state((state[0], state[1])),
)

plot_agent_actions_2d_seperated(
    states,
    agent,
    n_action,
    n_policy,
    save_path=f"{images_dir}",
    plot=args.plot,
    should_plot=lambda state: env.unwrapped._is_valid_state((state[0], state[1])),
)

# Only print w_values for valid states and save to CSV
valid_w_values = {state: values for state, values in w_values.items() if env.unwrapped._is_valid_state((state[0], state[1]))}

# First determine max length of valuez arrays
len_w_values_data = 0
for state, values in w_values.items():
    for obj_idx, valuez in values.items():
        len_w_values_data = max(len_w_values_data, len(valuez))

# Create empty array to store values
w_values_array = np.zeros((len(w_values) * len_w_values_data, n_policy))

# Fill array with values
row_idx = 0
for state, values in w_values.items():
    for t in range(len_w_values_data):
        for obj_idx in range(n_policy):
            if obj_idx in values and t < len(values[obj_idx]):
                w_values_array[row_idx, obj_idx] = values[obj_idx][t]
        row_idx += 1

# Create index labels for state and time
index_labels = []
for state in w_values.keys():
    for t in range(len_w_values_data):
        index_labels.append(f"state({state[0]},{state[1]})_t{t}")

# Create column names for objectives
column_names = [f"objective_{i}" for i in range(n_policy)]

# Create DataFrame with timesteps as rows and state-objective pairs as columns
w_values_df = pd.DataFrame(w_values_array, columns=column_names)
w_values_df.index.name = "timestep"

# Now you can save it without specifying columns again
w_values_df.to_csv(f"{results_dir}/w_values.csv", index=False)
print(f"saved w_values to {results_dir}/w_values.csv")

plot_agent_w_values(states, valid_w_values, n_policy, save_path=f"{images_dir}", plot=args.plot)

env.close()

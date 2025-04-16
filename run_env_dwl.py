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
import envs.mountain_car
from exploration_strategy.utils import create_exploration_strategy
from utils import extract_kwargs, build_parser, plot_over_time_multiple_subplots, smooth, kwargs_to_string
from agents import get_agent
from utils import generate_file_structure, kwargs_to_string

parser = build_parser()
parser.add_argument("--env", type=str, required=True, help="the environment to run")
parser.add_argument("--env_kwargs", type=str, nargs="*", help="environment kwargs")
parser.add_argument("--w_exploration", type=str, default=None, help="w_value exploration strategy for deciding which action to take")
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
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
exploration_strategy = create_exploration_strategy(args.exploration, **extract_kwargs(args.exploration_kwargs))
model_kwargs = extract_kwargs(args.model_kwargs)
model_kwargs["device"] = device
model_kwargs["input_shape"] = env.observation_space.shape
model_kwargs["num_actions"] = n_action
model_kwargs["num_policies"] = n_policy
model_kwargs["exploration_strategy"] = exploration_strategy
w_exploration_strategy = None
if args.w_exploration is not None:
    w_exploration_strategy = create_exploration_strategy(args.w_exploration, **extract_kwargs(args.w_exploration_kwargs))
    model_kwargs["w_exploration_strategy"] = w_exploration_strategy
agent_name = " ".join(args.model.split(" ")).lower()
agent = get_agent(agent_name, **model_kwargs)


if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    agent.load(args.path_to_load_model)

w_string = ""
w_exploration = ""
if w_exploration_strategy is not None:
    w_string = kwargs_to_string(w_exploration_strategy.info())
    w_exploration = args.w_exploration


results_dir, images_dir, models_dir, videos_dir = generate_file_structure(
    args.env,
    kwargs_to_string(env_kwargs),
    args.model,
    kwargs_to_string(args.model_kwargs),
    args.exploration,
    kwargs_to_string(exploration_strategy.info()),
    w_exploration,
    w_string,
)
results_path = f"{results_dir}/results.csv"
start_episode = 0

df = pd.DataFrame(columns=["episode", "episode_reward", "loss"])
if args.path_to_load_model is not None and len(args.path_to_load_model) > 0:
    prev_results_path = "".join(args.path_to_load_model.split("models/") + ["results.csv"]).replace(" ", "")
    if os.path.exists(prev_results_path):
        df = pd.read_csv(prev_results_path)
        start_episode = df.iloc[-1]["episode"] + 1

if args.w_check_interval != 0:
    low = env.observation_space.low
    high = env.observation_space.high
    x_step, y_step = 1, 1
    if env.observation_space.dtype == np.float32:
        x_step = (high[0] - low[0]) / 20
        y_step = (high[1] - low[1]) / 20

    rows = np.arange(low[1], high[1], y_step)
    cols = np.arange(low[0], high[0], x_step)
    states = []
    for row in rows:
        state_row = []
        for col in cols:
            if "deep-sea-treasure" in args.env:
                state_row.append(np.array([col, row]))
            else:
                state_row.append(np.array([row, col]))
        states.append(state_row)

    w_values: dict[tuple[int, int], dict[int, list[np.float64]]] = {tuple(state): {k: [] for k in range(n_policy)} for row in states for state in row}
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
            # print(obs, action, obs_, reward, done)
            agent.store_memory(obs, action, reward, obs_, done, info)
            obs = obs_
            for j in range(n_policy):
                episode_reward[j] = episode_reward[j] + reward[j]

        loss_info = agent.get_loss_values()
        df.loc[len(df)] = [i, episode_reward, loss_info]

        if i % args.print_interval == 0:
            print(f"Episode {i} completed  Reward {episode_reward}  Loss {loss_info}")

        if args.w_check_interval != 0 and i % args.w_check_interval == 0:
            for row in states:
                for state in row:
                    if env.unwrapped._is_valid_state(state):
                        cur_s_w_values = agent.get_all_info(state)[0]
                        for obj_idx, w_value in enumerate(cur_s_w_values):
                            w_values[tuple(state)][obj_idx].append(np.float64(w_value))
        agent.train()
        agent.update_params()
    print(f"Episode {num_episodes} completed")
except KeyboardInterrupt as e:
    response = input("Do you want to save the current results? (Y/n): ")
    if response.lower() == "n":
        print("Aborting...")
        exit(1)
    else:
        print("Saving results and finishing...")
        pass

agent.save(f"{models_dir}")

df.to_csv(results_path, index=False)


def parse_loss_string(s):
    try:
        tuples = s.strip("[]").split("),")
        tuples = [t.strip().strip("()") for t in tuples]
        return np.array([tuple(float(x.strip()) if x.strip() != "nan" else np.nan for x in t.split(",")) for t in tuples])
    except:
        return np.array([(np.nan, np.nan), (np.nan, np.nan)])


df = pd.read_csv(results_path)
loss_arrays = df["loss"].apply(parse_loss_string).values

q_loss, w_loss = [], []
for x in loss_arrays:
    if len(x) > 1:
        q_loss.append([x[i][0] for i in range(len(x))])
        w_loss.append([x[i][1] for i in range(len(x))])
    else:
        q_loss.append([x[0][i] for i in range(len(x[0]))])
        w_loss.append([float("nan") for _ in range(len(x[0]))])

q_loss = np.array(q_loss).T
w_loss = np.array(w_loss).T

plot_over_time_multiple_subplots(n_policy, q_loss, save_path=f"{images_dir}/q_loss.png", plot=args.plot)

plot_over_time_multiple_subplots(n_policy, w_loss, save_path=f"{images_dir}/w_loss.png", plot=args.plot)


def parse_list_string(s):
    try:
        s = s.strip("[]").split(",")
        return np.array([float(x) for x in s])
    except:
        return np.array([np.nan, np.nan])


episode_rewards_arrays = df["episode_reward"].apply(parse_list_string).values
episode_rewards = np.array([x for x in episode_rewards_arrays]).T

plot_over_time_multiple_subplots(n_policy, episode_rewards, save_path=f"{images_dir}/rewards.png", plot=args.plot)

window_size = 50
smoothed_rewards = []
for i in range(n_policy):
    smoothed_rewards.append(smooth(episode_rewards[i], window_size))
plot_over_time_multiple_subplots(n_policy, smoothed_rewards, save_path=f"{images_dir}/smoother_rewards.png", plot=args.plot)

if args.w_check_interval != 0:
    valid_w_values = {state: values for state, values in w_values.items() if env.unwrapped._is_valid_state((state[0], state[1]))}

    len_w_values_data = 0
    for state, values in valid_w_values.items():
        for obj_idx, valuez in values.items():
            len_w_values_data = max(len_w_values_data, len(valuez))

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

    column_names = ["state", "timestep"] + [f"objective_{i}" for i in range(n_policy)]

    w_values_df = pd.DataFrame(w_values_array, columns=column_names)
    w_values_df.index.name = "timestep"

    w_values_df.to_csv(f"{results_dir}/w_values.csv", index=False)

    plot_agent_w_values(states, valid_w_values, n_policy, save_path=f"{images_dir}", plot=args.plot)


env.close()

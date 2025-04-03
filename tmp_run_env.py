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

if args.record:
    env = RecordVideo(env, videos_dir, episode_trigger=lambda e: e % interval == 0)

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

    agent.train()
    agent.update_params()
print(f"Episode {i} completed")

agent.save(f"{models_dir}")

loss = [row[2] for row in csv_data]
episode_rewards = [row[1] for row in csv_data]
headers = ["episode", "episode_reward", "loss"]
df = pd.DataFrame(csv_data, columns=headers)
df.to_csv(f"{results_dir}/results.csv", index=False)

# performance measurements
loss = np.array(loss).T
plot_over_time_multiple_subplots(n_policy, loss, save_path=f"{images_dir}/loss.png", plot=args.plot)

episode_rewards = np.array(episode_rewards).T
plot_over_time_multiple_subplots(n_policy, episode_rewards, save_path=f"{images_dir}/rewards.png", plot=args.plot)

window_size = 50
smoothed_rewards = [None] * len(episode_rewards)
for i, episode_reward_objective in enumerate(episode_rewards):
    smoothed_rewards[i] = smooth(episode_reward_objective, window_size)
plot_over_time_multiple_subplots(n_policy, smoothed_rewards, save_path=f"{images_dir}/smoother_rewards.png", plot=args.plot)


env.close()

import gymnasium as gym
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from gym.wrappers.record_video import RecordVideo
import numpy as np
import torch
from agents.dueling_dqn_agent_4ly import DUELING_DQN
from agents.dqn_agent_4ly import DQN

gym.pprint_registry()

# The highway reward vec is [speed, right lane, collision]
# the rward space is [(0,1),(0,1),(-1,0)]
env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=2000)
env = RecordVideo(env, "videos/demo", episode_trigger=lambda e: e % 1000 == 0, name_prefix="video-cartpole")
# this is needed so that the video records smoothly :shrug:
# env.unwrapped.set_record_video_wrapper(env)
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
num_episodes = 100_000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = DQN(env.observation_space.shape, n_action, device=device, batch_size=64, learning_rate=0.1)

for i in range(num_episodes + 1):
    if i % 1000 == 0:
        print("Episode: ", i)
    episode_reward = 0
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action = agent.get_action(obs)
        obs_, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        agent.store_memory(obs, action, reward, obs_, done)
        agent.train()

agent.save_net("./agents/savedNets/demo/first_dqn_CartPole")

env.close()

import numpy as np
from gym import Env
from agents import AbstractAgent


def run_env(num_episodes: int, env: Env, agent: AbstractAgent, n_policy: int):
    """Runs the environment and agent for the specified number of epsiodes and returns the rewards, loss, and csv_data"""
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
        agent.train()
        agent.update_params()

    return episode_rewards, loss, csv_data

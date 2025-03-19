import numpy as np
import torch

from agents.democratic import DemocraticDQN


class ScaledDemocraticDQN(DemocraticDQN):
    def softmax(self, x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu()
            x = x.numpy()
        x = np.asarray(x).flatten()
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_action(self, x):
        """Get the action from the scaled democratic DQN"""
        action_advantages = np.array([0.0] * self.num_actions)

        for i, agent in enumerate(self.agents):
            q_values = agent.get_actions(x)
            scaled_q_values = self.softmax(q_values)
            preference_weighted_scaled_q_values = scaled_q_values * self.human_preference[i]
            action_advantages = action_advantages + preference_weighted_scaled_q_values

        return self.exploration_strategy.get_action(action_advantages, x), {}

    def get_actions(self, x):
        """Get every action from every agent"""
        action_advantages = np.array([0.0] * self.num_actions)

        for i, agent in enumerate(self.agents):
            q_values = agent.get_actions(x)
            scaled_q_values = self.softmax(q_values)
            preference_weighted_scaled_q_values = scaled_q_values * self.human_preference[i]
            action_advantages = action_advantages + preference_weighted_scaled_q_values

        return action_advantages

    def get_objective_info(self, x):
        """This is used to get info from each agent regarding the state x"""
        state_values = []

        for i, agent in enumerate(self.agents):
            q_values = agent.get_actions(x)
            scaled_q_values = self.softmax(q_values)
            state_values.append(scaled_q_values)

        return state_values

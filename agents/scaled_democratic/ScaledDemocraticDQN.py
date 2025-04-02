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

    def get_actions(self, x, human_preference: np.ndarray | None = None):
        """Get every action from every agent"""
        action_advantages = [None] * len(self.agents)

        for i, agent in enumerate(self.agents):
            q_values = np.array(agent.get_actions(x))
            scaled_q_values = self.softmax(q_values)
            action_advantages[i] = scaled_q_values

        return self.scalarization.scalarize(action_advantages, human_preference)

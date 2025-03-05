import numpy as np
from .BaseDDWN import BaseDDWN


class SameActionDDWN(BaseDDWN):
    """
    Same Action DDWN
    This class modifies the BaseDDWN to only update the W-network for models whos maximum Q-values are not the same as the selected action.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_action(self, x) -> tuple[int, dict]:
        """Get the action nomination for the given state"""
        q_values = []
        w_values = []
        for agent in self.agents:
            agent_q_values = agent.get_actions(x)
            agent_w_value = agent.get_w_value(x)
            q_values.append(agent_q_values)
            w_values.append(agent_w_value)

        weighted_actions = self.get_weighted_actions(q_values, w_values)

        action_selected = self.exploration_strategy.get_action(weighted_actions, x)
        return action_selected, {"q_values": q_values, "w_values": w_values}

    def get_weighted_actions(self, q_values: list[list[float]], w_values: list[float]) -> list[float]:
        """Get the weighted actions for the given Q-values and W-values"""
        weighted_actions = [0.0] * self.num_actions
        for agent_q_values, agent_w_value in zip(q_values, w_values):
            weighted_actions = weighted_actions + (self.softmax(agent_q_values) * agent_w_value)
        return weighted_actions

    def store_memory(self, s, a, rewards, s_, d, info: dict):
        for i in range(self.num_policies):
            self.agents[i].store_memory(s, a, rewards[i], s_, d)
            if np.argmax(info["q_values"][i]) != a:
                # check if the agents maximum action is the same as the selected action only update W-network for this agent if it is not the selected action
                self.agents[i].store_w_memory(s, a, rewards[i], s_, d)

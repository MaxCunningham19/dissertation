import numpy as np
from .DemocraticDWL import DemocraticDWL


class DemocraticDWL_MaxActionMaxQValue(DemocraticDWL):

    def get_action(self, x) -> tuple[int, dict]:
        """Get the action nomination for the given state"""
        actions_values = np.array([0.0] * self.num_actions)
        max_actions = []
        for q_agent, w_agent in self.agents:
            q_values = np.array(q_agent.get_actions(x))
            max_actions.append(np.argmax(q_values))
            w_value = w_agent.get_value(x)
            actions_values = actions_values + q_values * w_value

        action = self.exploration_strategy.get_action(actions_values, x)
        max_q_value = float("-inf")
        max_policy = None
        for i in range(len(max_actions)):
            if max_actions[i] == action:
                if q_values[i] > max_q_value:
                    max_q_value = q_values[i]
                    max_policy = i
        return action, {"max_policy": max_policy}

    def store_memory(self, s, a, rewards, s_, d, info: dict):
        for i in range(self.num_policies):
            q_agent, w_agent = self.agents[i]
            q_agent.store_memory(s, a, rewards[i], s_, d)
            # dont update the w_agent if its max action is selected and its q_value is the max q_value of the models with the same max action
            if i == info["max_policy"]:
                w_agent.store_memory(s, a, rewards[i], s_, d)

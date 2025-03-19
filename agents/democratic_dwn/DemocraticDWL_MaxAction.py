import numpy as np
from .DemocraticDWL import DemocraticDWL


class DemocraticDWL_MaxAction(DemocraticDWL):

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
        maximum_policies = []
        for i in range(len(max_actions)):
            if max_actions[i] == action:
                maximum_policies.append(i)

        return action, {"max_policies": maximum_policies}

    def store_memory(self, s, a, rewards, s_, d, info: dict):
        for i in range(self.num_policies):
            q_agent, w_agent = self.agents[i]
            q_agent.store_memory(s, a, rewards[i], s_, d)
            # only store agents whos maximum action did not coincide with the selected action
            if i not in info["max_policies"]:
                w_agent.store_memory(s, a, rewards[i], s_, d)

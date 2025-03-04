import numpy as np
from agents import AbstractAgent
from utils import plot_agent_actions_2d


class agent(AbstractAgent):
    def __init__(self, n_policy: int, n_action: int):
        self.n_policy = n_policy
        self.n_action = n_action

    def get_action(self, x):
        return 0, {}

    def store_memory(self, s, a, rewards, s_, d, info: dict) -> None:
        pass

    def train(self) -> None:
        pass

    def update_params(self) -> None:
        pass

    def get_loss_values(self) -> list[tuple[float, ...]]:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def get_objective_info(self, x) -> list[list[float]]:
        return [np.random.uniform(0, 1, self.n_action) for _ in range(self.n_policy)]


for num_policy in range(5):
    for num_action in range(5):
        _agent = agent(num_policy, num_action)
        states = [[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]], [[2, 0], [2, 1], [2, 2]]]
        plot_agent_actions_2d(states, _agent, n_action=num_action, n_policy=num_policy, save_path=f"{num_policy}o{num_action}a.png")

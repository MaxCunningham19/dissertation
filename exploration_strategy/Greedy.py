import numpy as np

from .ExplorationStrategy import ExplorationStrategy


class Greedy(ExplorationStrategy):
    def _get_action(self, actions: np.ndarray, state=None):
        return np.argmax(actions)

    def update_parameters(self):
        return

    def info(self):
        return ""

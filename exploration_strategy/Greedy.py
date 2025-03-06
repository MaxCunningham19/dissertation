import numpy as np

from .ExplorationStrategy import ExplorationStrategy


class Greedy(ExplorationStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_action(self, actions: np.ndarray, state=None):
        return np.argmax(actions)

    def _update_parameters(self):
        return

    def info(self):
        return ""

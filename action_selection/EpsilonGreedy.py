import numpy as np

from .ExplorationStrategy import ExplorationStrategy


class EpsilonGreedy(ExplorationStrategy):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def _get_action(self, actions: np.ndarray, state=None):
        if np.random.uniform() > self.epsilon:
            return np.argmax(actions)

        return np.random.randint(0, len(actions))

    def update_parameters(self):
        return

    def info(self):
        return self.epsilon

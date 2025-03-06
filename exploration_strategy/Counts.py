from typing import Callable, Any, Hashable
import numpy as np

from .ExplorationStrategy import ExplorationStrategy
from utils import softmax


class Counts(ExplorationStrategy):
    def __init__(self, alpha=1.0, beta=2.0, state_map: Callable[[Any | None], Hashable] = lambda x: tuple(x), **kwargs):
        super().__init__(**kwargs)
        self.counts = {}
        self.state_map = state_map
        self.alpha = alpha
        self.beta = beta

    def _get_action(self, actions: np.ndarray, state=None):
        state = self.state_map(state)

        if state not in self.counts:
            action = np.argmin(actions)
            self.counts[state] = np.array([1.0] * len(actions))
        else:
            counts_state = np.array(self.counts[state])
            probabilities = softmax(((actions) ** self.alpha) / ((counts_state) ** self.beta))
            action = np.random.choice(len(actions), p=probabilities)

        self.counts[state][action] = self.counts[state][action] + 1.0

        return action

    def _update_parameters(self):
        return

    def info(self):
        return self.counts, self.alpha, self.beta

import numpy as np

from EpsilonGreedy import EpsilonGreedy


class DecayEpsilonGreedy(EpsilonGreedy):
    def __init__(self, epsilon=0.9, epsilon_decay=0.9, epsilon_min=0.1):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def update_parameters(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

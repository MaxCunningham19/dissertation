import numpy as np

from .ExplorationStrategy import ExplorationStrategy

MIN_TEMPERATURE = 1e-6  # Prevents divby zero


class Boltzmann(ExplorationStrategy):
    def __init__(self, temperature=1.0):
        self.temperature = max(MIN_TEMPERATURE, temperature)

    def _get_action(self, actions: np.ndarray, state=None):
        stable_exp = actions / self.temperature
        stable_exp = stable_exp - np.max(stable_exp)
        probabilities = stable_exp / np.sum(stable_exp)
        return np.random.choice(len(actions), p=probabilities)

    def update_parameters(self):
        return

    def info(self):
        return self.temperature

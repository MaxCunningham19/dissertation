import numpy as np

from .ExplorationStrategy import ExplorationStrategy

MIN_TEMPERATURE = 1e-6  # Prevents divby zero


class Boltzmann(ExplorationStrategy):
    def __init__(self, temperature=1.0):
        self.temperature = max(MIN_TEMPERATURE, temperature)

    def _get_action(self, actions: np.ndarray, state=None):
        # The issue is we're not exponentiating the values
        # Without exp(), dividing by temperature and subtracting max just makes larger values smaller
        # and smaller values relatively larger, inverting the preferences
        stable_exp = np.exp(actions / self.temperature)
        stable_exp = stable_exp / np.sum(stable_exp)  # Normalize to get probabilities
        return np.random.choice(len(actions), p=stable_exp)

    def update_parameters(self):
        return

    def info(self):
        return self.temperature

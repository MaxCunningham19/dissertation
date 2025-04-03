import numpy as np


from .ExplorationStrategy import ExplorationStrategy

from exploration_strategy.softmax import softmax

MIN_TEMPERATURE = 1e-6  # Prevents divby zero


class Boltzmann(ExplorationStrategy):
    def __init__(self, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = max(MIN_TEMPERATURE, temperature)

    def _get_action(self, actions: np.ndarray, state=None):
        stable_exp = softmax((actions - np.max(actions)) / self.temperature)
        return np.random.choice(len(actions), p=stable_exp)

    def _update_parameters(self):
        return

    def info(self) -> dict:
        return {"temperature": self.temperature}

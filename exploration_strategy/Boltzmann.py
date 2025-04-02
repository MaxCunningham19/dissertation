import numpy as np

from utils.utils import softmax

from .ExplorationStrategy import ExplorationStrategy

MIN_TEMPERATURE = 1e-6  # Prevents divby zero


class Boltzmann(ExplorationStrategy):
    def __init__(self, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = max(MIN_TEMPERATURE, temperature)

    def _get_action(self, actions: np.ndarray, state=None):

        dist_from_max = actions - np.max(actions)
        e_x = np.exp((dist_from_max) / self.temperature)
        stable_exp = e_x / np.sum(e_x)  # Normalize to get probabilities
        return np.random.choice(len(actions), p=stable_exp)

    def _update_parameters(self):
        return

    def info(self) -> dict:
        return {"temperature": self.temperature}

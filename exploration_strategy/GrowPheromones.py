from typing import Callable, Any, Hashable

from .Pheromones import Pheromones

MIN_PHEROMONE = 1.0
NON_ZERO = 1e-6


class GrowPheromones(Pheromones):
    def __init__(
        self,
        pheromone_decay=0.7,
        pheromone_inc=1.0,
        alpha=1.0,
        beta=2.0,
        state_map: Callable[[Any | None], Hashable] = lambda x: tuple(x),
        pheromone_min=MIN_PHEROMONE,
        alpha_max=5.0,
        alpha_growth=1.01,
        beta_decay=0.99,
        beta_min=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pheromone_decay = pheromone_decay
        self.pheromone_inc = pheromone_inc
        self.pheromone_min = max(NON_ZERO, pheromone_min)
        self.pheromones = {}
        self.state_map = state_map
        self.beta_decay = beta_decay
        self.alpha = alpha
        self.beta = beta
        self.alpha_max = alpha_max
        self.alpha_growth = alpha_growth
        self.beta_min = beta_min

    def _update_parameters(self):
        self.alpha = min(self.alpha * self.alpha_growth, self.alpha_max)
        self.beta = max(self.beta * self.beta_decay, self.beta_min)
        self.force_update_parameters()

    def force_update_parameters(self):
        for state in self.pheromones:
            pheromones = self.pheromones[state]
            for i in range(len(pheromones)):
                pheromones[i] = max(self.pheromone_min, pheromones[i] * self.pheromone_decay)
            self.pheromones[state] = pheromones

    def info(self) -> dict:
        return {
            "pheromone_decay": self.pheromone_decay,
            "pheromone_inc": self.pheromone_inc,
            "alpha": self.alpha,
            "alpha_max": self.alpha_max,
            "alpha_growth": self.alpha_growth,
            "beta": self.beta,
            "beta_decay": self.beta_decay,
            "beta_min": self.beta_min,
        }

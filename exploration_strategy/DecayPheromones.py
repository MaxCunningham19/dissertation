from typing import Callable, Any, Hashable

from .Pheromones import Pheromones

MIN_PHEROMONE = 1.0
NON_ZERO = 1e-6


class DecayPheromones(Pheromones):
    def __init__(
        self,
        pheromone_decay=0.7,
        pheromone_inc=1.0,
        alpha=1.0,
        beta=5.0,
        beta_decay=0.9,
        state_map: Callable[[Any | None], Hashable] = lambda x: tuple(x),
        pheromone_min=MIN_PHEROMONE,
        beta_min=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pheromone_decay = pheromone_decay
        self.pheromone_inc = pheromone_inc
        self.pheromone_min = max(NON_ZERO, pheromone_min)
        self.pheromones = {}
        self.state_map = state_map
        self.alpha = alpha
        self.beta = beta
        self.beta_decay = beta_decay
        self.beta_min = beta_min

    def info(self) -> dict:
        return {
            "pheromone_decay": self.pheromone_decay,
            "pheromone_inc": self.pheromone_inc,
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_decay": self.beta_decay,
            "beta_min": self.beta_min,
        }

    def _update_parameters(self):
        self.beta = max(self.beta * self.beta_decay, self.beta_min)
        self.force_update_parameters()

    def force_update_parameters(self):
        for state in self.pheromones:
            pheromones = self.pheromones[state]
            for i in range(len(pheromones)):
                pheromones[i] = max(self.pheromone_min, pheromones[i] * self.pheromone_decay)
            self.pheromones[state] = pheromones

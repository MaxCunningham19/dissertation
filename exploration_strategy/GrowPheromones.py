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
        self.alpha_max = alpha_max
        self.alpha_growth = alpha_growth

    def _update_parameters(self):
        self.alpha = min(self.alpha * self.alpha_growth, self.alpha_max)
        self.force_update_parameters()

    def force_update_parameters(self):
        for state in self.pheromones:
            pheromones = self.pheromones[state]
            for i in range(len(pheromones)):
                pheromones[i] = max(self.pheromone_min, pheromones[i] * self.pheromone_decay)
            self.pheromones[state] = pheromones

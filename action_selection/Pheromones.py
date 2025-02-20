from typing import Callable, Any, Hashable
import numpy as np

from ExplorationStrategy import ExplorationStrategy

MIN_PHEROMONE = 1.0
NON_ZERO = 1e-6


class Pheromones(ExplorationStrategy):
    def __init__(
        self,
        pheromone_decay=0.5,
        pheromone_inc=1.0,
        alpha=1.0,
        beta=1.0,
        state_map: Callable[[Any | None], Hashable] = lambda x: x,
        pheromone_min=MIN_PHEROMONE,
    ):
        self.pheromone_decay = pheromone_decay
        self.pheromone_inc = pheromone_inc
        self.pheromone_min = max(NON_ZERO, pheromone_min)
        self.pheromones = {}
        self.state_map = state_map
        self.alpha = alpha
        self.beta = beta

    def _get_action(self, actions: np.ndarray, state=None):
        state = self.state_map(state)

        if not state in self.pheromones:  # add state to pheromone map
            action = np.argmax(actions)
            self.pheromones[state] = np.array([self.pheromone_min] * self.num_actions)
        else:
            pheromones_state = np.array(self.pheromones[state])
            probabilities = self.softmax(((actions) ** self.alpha) / ((pheromones_state) ** self.beta))
            action = np.random.choice(self.num_actions, p=probabilities)

        self.pheromones[state][action] = self.pheromones[state][action] + self.pheromone_inc

        return action

    def update_parameters(self):
        for state in self.pheromones:
            pheromones = self.pheromones[state]
            for i in range(len(pheromones)):
                pheromones[i] = max(self.pheromone_min, pheromones[i] * self.pheromone_decay)
            self.pheromones[state] = pheromones

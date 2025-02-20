import numpy as np
from abc import ABC, abstractmethod


class ExplorationStrategy(ABC):

    def get_action(self, actions: np.ndarray, state=None):
        """
        Returns the index of the action
        """
        if type(actions).__name__ != "ndarray":
            return -1

        return self._get_action(actions, state)

    @abstractmethod
    def _get_action(self, actions: np.ndarray, state):
        """
        Returns the index of the action
        """
        pass

    @abstractmethod
    def update_parameters(self):
        """
        Updates heuristic parameters
        """
        pass

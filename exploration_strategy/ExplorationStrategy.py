import numpy as np
from abc import ABC, abstractmethod

import torch


class ExplorationStrategy(ABC):

    def get_action(self, actions: np.ndarray | torch.Tensor, state=None):
        """
        Returns the index of the action
        """
        if isinstance(actions, torch.Tensor):
            if actions.is_cuda:
                actions = actions.cpu()
            actions = actions.numpy()

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

    @abstractmethod
    def info(self):
        """Returns the current parameters of the strategy"""
        pass

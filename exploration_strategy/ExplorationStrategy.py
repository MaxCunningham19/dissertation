import numpy as np
from abc import ABC, abstractmethod

import torch


class ExplorationStrategy(ABC):
    def __init__(self, delay_updates=0, **kwargs):
        super().__init__(**kwargs)
        self.delay_updates = delay_updates
        self.update_counts = 0

    def get_action(self, actions: np.ndarray | torch.Tensor, state=None):
        """
        Returns the index of the action
        """
        if isinstance(actions, torch.Tensor):
            if actions.is_cuda:
                actions = actions.cpu()
            actions = actions.numpy()
        actions = np.array(actions).flatten()
        if type(actions).__name__ != "ndarray":
            return -1

        return self._get_action(actions, state)

    @abstractmethod
    def _get_action(self, actions: np.ndarray, state):
        """
        Returns the index of the action
        """
        pass

    def update_parameters(self):
        """
        Updates heuristic parameters
        """
        if self.update_counts < self.delay_updates:
            self.force_update_parameters()
            self.update_counts += 1
            # print(f"Delaying update for {self.delay_updates - self.update_counts} more updates")
            return
        self._update_parameters()

    @abstractmethod
    def _update_parameters(self):
        """
        Updates heuristic parameters
        """
        pass

    @abstractmethod
    def info(self) -> dict:
        """Returns the current parameters of the strategy"""
        pass

    def force_update_parameters(self):
        """
        Forces the update of the parameters
        """
        pass

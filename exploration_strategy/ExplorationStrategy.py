import json
import os
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

    def save(self, path: str) -> None:
        """Save the exploration strategy"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        info_dict = self.info()
        info_dict["strategy_name"] = self.__class__.__name__
        try:
            with open(path, "w") as f:
                json.dump(info_dict, f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error saving exploration strategy to {path}: {str(e)}")

    def load(self, path: str) -> None:
        """Load the exploration strategy"""
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                info_dict = json.load(f)
            strategy_name = info_dict.pop("strategy_name")
            if self.__class__.__name__ == strategy_name:
                self.__init__(**info_dict)
            else:
                print(
                    f"Warning: The exploration strategy {self.__class__.__name__} was passed the class but in the file {path} the stored strategy is {strategy_name}.\n"
                    + " Continuing with the passed strategy."
                )
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return

    def __str__(self) -> str:
        return self.__class__.__name__

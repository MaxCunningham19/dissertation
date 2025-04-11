from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ActionScalarization(ABC):
    @staticmethod
    @abstractmethod
    def scalarize(action_values: np.ndarray, human_preference: Optional[np.ndarray]) -> np.ndarray:
        """
        Scalarize action values using human preferences.

        Args:
            action_values: 2D array of shape (n_objectives, n_actions)
            human_preference: 1D array of shape (n_objectives,)

        Returns:
            np.ndarray: Scalarized action values
        """
        pass


class LinearScalarization(ActionScalarization):
    """
    Scalarize action values using the linear scalarization method.

    This method computes the weighted sum of the action values.
    """

    @staticmethod
    def scalarize(action_values: np.ndarray, human_preference: Optional[np.ndarray]) -> np.ndarray:
        if human_preference is None:
            return np.sum(action_values, axis=0)
        return np.dot(human_preference, action_values)


class ChebyshevScalarization(ActionScalarization):
    """
    Scalarize action values using the Chebyshev scalarization method.

    This method maximizes the minimum weighted deviation from the ideal rewards for each action.
    """

    @staticmethod
    def scalarize(action_values: np.ndarray, human_preference: Optional[np.ndarray]) -> np.ndarray:
        if human_preference is None:
            human_preference = np.ones(action_values.shape[0])
        weights = np.array(human_preference)
        num_objectives, num_actions = len(action_values), len(action_values[0])
        ideal_rewards = np.max(action_values, axis=1)  # ideal reward for each objective

        scalarized_rewards = np.zeros(num_actions)
        for action in range(num_actions):
            deviations = np.abs(action_values[:, action] - ideal_rewards)
            weighted_deviations = human_preference * deviations
            scalarized_rewards[action] = np.max(weighted_deviations)

        return -scalarized_rewards


SCALARIZATION_METHODS = {"linear": LinearScalarization, "chebyshev": ChebyshevScalarization}


def get_scalarization_method(method_name: str) -> ActionScalarization:
    scalarization_method = SCALARIZATION_METHODS.get(method_name, None)
    if scalarization_method is None:
        raise ValueError(f"Invalid scalarization method: {method_name}, must be one of {list(SCALARIZATION_METHODS.keys())}")
    return scalarization_method

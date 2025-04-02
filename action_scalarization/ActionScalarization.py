from abc import ABC, abstractmethod

import numpy as np


class ActionScalarization(ABC):
    @staticmethod
    @abstractmethod
    def scalarize(action_values: np.ndarray, human_preference: np.ndarray) -> float:
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
    def scalarize(action_values: np.ndarray, human_preference: np.ndarray) -> np.ndarray:
        return np.dot(action_values, human_preference)


class ChebyshevScalarization(ActionScalarization):
    """
    Scalarize action values using the Chebyshev scalarization method.

    This method maximizes the minimum weighted deviation from the ideal rewards for each action.
    """

    @staticmethod
    def scalarize(action_values: np.ndarray, human_preference: np.ndarray) -> np.ndarray:
        weights = np.array(human_preference)

        ideal_rewards = np.max(action_values, axis=1)

        deviations = weights[:, None] * np.abs(ideal_rewards[:, None] - action_values)
        scalarized_rewards = np.max(deviations, axis=0)

        return -scalarized_rewards


def get_scalarization_method(method_name: str) -> ActionScalarization:
    if method_name == "linear":
        return LinearScalarization
    elif method_name == "chebyshev":
        return ChebyshevScalarization
    else:
        raise ValueError(f"Invalid scalarization method: {method_name}")

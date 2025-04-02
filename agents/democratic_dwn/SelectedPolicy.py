from abc import ABC, abstractmethod
import numpy as np


class SelectedPolicy(ABC):
    @staticmethod
    @abstractmethod
    def select_policy(objective_action_values: np.ndarray, action_values: np.ndarray, w_values: np.ndarray, selected_action: int) -> list[int]:
        pass


class MaxActionValue(SelectedPolicy):
    """This returns the policy whos action value is the largest of the selected actions"""

    @staticmethod
    def select_policy(objective_action_values: np.ndarray, action_values: np.ndarray, w_values: np.ndarray, selected_action: int) -> list[int]:
        selected_action_values = objective_action_values[:, selected_action]
        return np.where(selected_action_values == np.max(selected_action_values))[0].tolist()


class MaxWValue(SelectedPolicy):
    """This returns the policy whos w value is the largest"""

    @staticmethod
    def select_policy(objective_action_values: np.ndarray, action_values: np.ndarray, w_values: np.ndarray, selected_action: int) -> list[int]:
        return np.where(w_values == np.max(w_values))[0].tolist()


class MaxActionIsSelectedAction(SelectedPolicy):
    """This returns the policy whos max action is the selected action"""

    @staticmethod
    def select_policy(objective_action_values: np.ndarray, action_values: np.ndarray, w_values: np.ndarray, selected_action: int) -> list[int]:
        selected_policies = []
        for i in range(objective_action_values.shape[0]):
            if np.argmax(objective_action_values[i]) == selected_action:
                selected_policies.append(i)
        return selected_policies


class NoSelection(SelectedPolicy):
    """This returns no policies"""

    @staticmethod
    def select_policy(objective_action_values: np.ndarray, action_values: np.ndarray, w_values: np.ndarray, selected_action: int) -> list[int]:
        return np.arange(objective_action_values.shape[0]).tolist()


def get_selected_policy(selected_policy_name: str) -> SelectedPolicy:
    if selected_policy_name == "max_action_value":
        return MaxActionValue()
    elif selected_policy_name == "max_w_value":
        return MaxWValue()
    elif selected_policy_name == "selected_action_is_max_action":
        return MaxActionIsSelectedAction()
    elif selected_policy_name == "none":
        return NoSelection()
    else:
        raise ValueError(f"Selected policy {selected_policy_name} not found")

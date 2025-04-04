from abc import ABC, abstractmethod
import numpy as np


class AbstractAgent(ABC):
    """These are the minimum methods that are required for an agent to be ran in the environment"""

    @abstractmethod
    def get_action(self, x, human_preference: np.ndarray | None = None) -> tuple[int, dict]:
        """Get an action from the agent in state x"""
        pass

    @abstractmethod
    def get_actions(self, x, human_preference: np.ndarray | None = None) -> np.ndarray:
        """Get all action values from the agent in state x"""
        pass

    @abstractmethod
    def store_memory(self, s, a, rewards, s_, d, info: dict) -> None:
        """Store a memory in agent"""
        pass

    @abstractmethod
    def train(self) -> None:
        """Train the agent"""
        pass

    @abstractmethod
    def update_params(self) -> None:
        """Update the parameters of the agent"""
        pass

    @abstractmethod
    def get_loss_values(self) -> list[tuple[float, ...]]:
        """Get the loss values of the agent"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent"""
        pass

    @abstractmethod
    def get_objective_info(self, x, human_preference: np.ndarray | None = None) -> list[np.ndarray]:
        """Get all action values in state x for each objective"""
        pass

from typing import Literal
import numpy as np
import torch

from action_scalarization import ActionScalarization
from agents.democratic_dwn import DemocraticDWL
from exploration_strategy import ExplorationStrategy
from exploration_strategy.utils import create_exploration_strategy
from utils.utils import l1_normalization, softmax


class ScaledDemocraticDWL(DemocraticDWL):

    def __init__(
        self,
        input_shape,
        num_actions,
        num_policies,
        exploration_strategy: ExplorationStrategy,
        scalarization: ActionScalarization,
        w_normalization: Literal["L1", "Softmax"] = "Softmax",
        normalization: Literal["Softmax", "L1"] = "Softmax",
        memory_size=100000,
        batch_size=1024,
        learning_rate=0.001,
        gamma=0.9,
        hidlyr_nodes=256,
        init_learn_steps_num=128,
        beta_start=0.4,
        beta_inc=1.002,
        device=None,
        tau=0.001,
        w_tau=0.001,
        w_alpha=0.001,
        per_epsilon=0.001,
        seed=404,
    ):

        super().__init__(
            input_shape,
            num_actions=num_actions,
            num_policies=num_policies,
            exploration_strategy=create_exploration_strategy("greedy"),
            memory_size=memory_size,
            batch_size=batch_size,
            scalarization=scalarization,
            w_normalization=w_normalization,
            learning_rate=learning_rate,
            gamma=gamma,
            hidlyr_nodes=hidlyr_nodes,
            init_learn_steps_num=init_learn_steps_num,
            beta_start=beta_start,
            beta_inc=beta_inc,
            device=device,
            tau=tau,
            w_tau=w_tau,
            w_alpha=w_alpha,
            per_epsilon=per_epsilon,
            seed=seed,
        )
        self.exploration_strategy = exploration_strategy
        self.normalization = normalization

    def get_action_and_w_values(self, x, human_preference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the action nomination and the w-values for the given state
        Returns:
            action_values: np.ndarray, the action values for the given state
            w_values: np.ndarray, the w-values for the given state
            objective_action_values: np.ndarray,  action values for each objective
        """
        if human_preference is None:
            human_preference = np.ones(self.num_policies)
        w_values = self.get_w_values(x, human_preference)
        objective_action_values = np.zeros((self.num_policies, self.num_actions), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            objective_action_values[i] = self.normalize(agent.get_actions(x))
        action_values = self.scalarization.scalarize(objective_action_values, w_values)
        return action_values, w_values, objective_action_values

    def get_action(self, x, human_preference: np.ndarray | None = None) -> tuple[int, dict]:
        """Get the action nomination for the given state"""
        action_values, w_values, _ = self.get_action_and_w_values(x, human_preference)
        action_sel = self.exploration_strategy.get_action(action_values, x)
        return action_sel, {"policies_sel": None, "w_values": w_values}

    def get_actions(self, x, human_preference: np.ndarray | None = None) -> np.ndarray:
        """Get all action values from the agent in state x"""
        action_values, _, _ = self.get_action_and_w_values(x, human_preference)
        return action_values

    def store_memory(self, s, a, rewards, s_, d, info: dict) -> None:
        """Store a memory in agent"""
        for i in range(self.num_policies):
            agent = self.agents[i]
            if self.q_learning:
                agent.store_memory(s, a, rewards[i], s_, d)
            if self.w_learning:
                agent.store_w_memory(s, a, rewards[i], s_, d)

    def name(self) -> str:
        return f"Scaled Democratic DWL with {self.scalarization.name()}, {self.normalization}, {self.w_normalization}"

    def normalize(self, x) -> np.ndarray:
        if self.normalization == "L1":
            return l1_normalization(x)
        return softmax(x)

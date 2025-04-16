import numpy as np
import torch
from typing import Literal

from action_scalarization import ActionScalarization
from agents.democratic import DemocraticDQN
from agents.dqn_agent import DQN
from exploration_strategy import ExplorationStrategy
from utils.utils import l1_normalization, softmax


class ScaledDemocraticDQN(DemocraticDQN):
    def __init__(
        self,
        input_shape,
        num_actions,
        num_policies,
        exploration_strategy: ExplorationStrategy,
        scalarization: ActionScalarization,
        normalization: Literal["Softmax", "L1"] = "Softmax",
        batch_size=1024,
        memory_size=100000,
        learning_rate=0.001,
        gamma=0.9,
        tau=0.001,
        per_epsilon=0.001,
        beta_start=0.4,
        beta_inc=1.002,
        hidlyr_nodes=256,
        seed=404,
        device=None,
    ):

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_policies = num_policies
        self.num_states = self.input_shape[0]
        self.exploration_strategy = exploration_strategy
        self.scalarization = scalarization
        self.normalization = normalization

        # Construct Agents for each policy
        self.agents: list[DQN] = []
        for i in range(self.num_policies):
            self.agents.append(
                DQN(
                    input_shape=self.input_shape,
                    num_actions=self.num_actions,
                    batch_size=batch_size,
                    tau=tau,
                    memory_size=memory_size,
                    learning_rate=learning_rate,
                    exploration_strategy=None,
                    gamma=gamma,
                    per_epsilon=per_epsilon,
                    beta_start=beta_start,
                    beta_inc=beta_inc,
                    seed=seed,
                    device=device,
                    hidlyr_nodes=hidlyr_nodes,
                )
            )

    def normalize(self, x) -> np.ndarray:
        if self.normalization == "L1":
            return l1_normalization(x)
        return softmax(x)

    def get_actions(self, x, human_preference: np.ndarray | None = None):
        """Get every action from every agent"""
        action_advantages = [None] * len(self.agents)

        for i, agent in enumerate(self.agents):
            q_values = np.array(agent.get_actions(x))
            scaled_q_values = self.normalize(q_values)
            action_advantages[i] = scaled_q_values

        return self.scalarization.scalarize(np.array(action_advantages), human_preference)

    def name(self) -> str:
        return f"Scaled Democratic DQN with {self.scalarization.name()} Scalarization and {self.normalization} Normalization"

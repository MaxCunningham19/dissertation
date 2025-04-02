import numpy as np

from action_scalarization import ActionScalarization
from agents.democratic_dwn.SelectedPolicy import SelectedPolicy
from agents.dwn import DWL
from exploration_strategy.Greedy import Greedy
from exploration_strategy.DecayEpsilonGreedy import DecayEpsilonGreedy
from exploration_strategy import ExplorationStrategy
from agents.AbstractAgent import AbstractAgent
from ..dwn_agent import DWN
from ..dqn_agent import DQN


class DemocraticDWL(DWL):

    def __init__(
        self,
        input_shape,
        num_actions,
        num_policies,
        exploration_strategy: ExplorationStrategy,
        scalarization: ActionScalarization,
        selected_policy: SelectedPolicy,
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
        walpha=0.001,
        seed=404,
    ):

        super().__init__(
            input_shape,
            num_actions=num_actions,
            num_policies=num_policies,
            exploration_strategy=exploration_strategy,
            memory_size=memory_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            hidlyr_nodes=hidlyr_nodes,
            init_learn_steps_num=init_learn_steps_num,
            beta_start=beta_start,
            beta_inc=beta_inc,
            device=device,
            tau=tau,
            w_tau=w_tau,
            walpha=walpha,
            seed=seed,
        )
        self.scalarization = scalarization
        self.selected_policy = selected_policy

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
        action_values = np.zeros(self.num_actions, dtype=np.float32)
        objective_action_values = np.zeros((self.num_policies, self.num_actions), dtype=np.float32)
        for i in range(self.num_policies):
            action_values += np.array(self.agents[i].get_actions(x)) * w_values[i]
            objective_action_values[i] = np.array(self.agents[i].get_actions(x)) * w_values[i]

        return action_values, w_values, objective_action_values

    def policy_store_memory_selection(
        self, objective_action_values: np.ndarray, action_values: np.ndarray, w_values: np.ndarray, selected_action: int
    ) -> list[int]:
        """Select the policies to store memory for the given state"""
        return self.selected_policy.select_policy(objective_action_values, action_values, w_values, selected_action)

    def get_action(self, x, human_preference: np.ndarray | None = None) -> tuple[int, dict]:
        """Get the action nomination for the given state"""
        action_values, w_values = self.get_action_and_w_values(x, human_preference)
        action_sel = self.exploration_strategy.get_action(action_values, x)
        policy_sel = self.selected_policy.select_policy(action_values, w_values, action_sel)
        return action_sel, {"policies_sel": policy_sel, "w_values": w_values}

    def get_actions(self, x, human_preference: np.ndarray | None = None) -> np.ndarray:
        """Get all action values from the agent in state x"""
        if human_preference is None:
            human_preference = np.ones(self.num_policies)
        w_values = self.get_w_values(x, human_preference)
        action_values = np.zeros(self.num_actions, dtype=np.float32)
        for i in range(self.num_policies):
            action_values += np.array(self.agents[i].get_actions(x)) * w_values[i]

        return action_values

    def store_memory(self, s, a, rewards, s_, d, info: dict) -> None:
        """Store a memory in agent"""
        for i in range(self.num_policies):
            agent = self.agents[i]
            if self.q_learning:
                agent.store_memory(s, a, rewards[i], s_, d)
            if self.w_learning and i in info["policies_sel"]:  # Do not store experience of the policy we selected
                agent.store_w_memory(s, a, rewards[i], s_, d)

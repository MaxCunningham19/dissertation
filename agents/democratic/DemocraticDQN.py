import numpy as np
import os
import copy
from exploration_strategy import ExplorationStrategy
from agents.AbstractAgent import AbstractAgent
from ..dqn_agent import DQN
from action_scalarization import ActionScalarization


class DemocraticDQN(AbstractAgent):

    def __init__(
        self,
        input_shape,
        num_actions,
        num_policies,
        exploration_strategy: ExplorationStrategy,
        scalarization: ActionScalarization,
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
        human_preference=None,
    ):

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_policies = num_policies
        self.num_states = self.input_shape[0]
        self.exploration_strategy = exploration_strategy
        self.scalarization = scalarization
        self.human_preference = human_preference
        if self.human_preference is None or len(human_preference) != self.num_policies:
            self.human_preference = [1.0] * self.num_policies

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

    def get_action(self, x, human_preference: np.ndarray | None = None):
        """Get the action from the democratic DQN"""
        action_advantages = self.get_actions(x, human_preference)
        return self.exploration_strategy.get_action(action_advantages, x), {}

    def get_actions(self, x, human_preference: np.ndarray | None = None):
        """Get every action from every agent"""
        action_advantages = [None] * len(self.agents)

        for i, agent in enumerate(self.agents):
            q_values = np.array(agent.get_actions(x))
            action_advantages[i] = q_values

        return self.scalarization.scalarize(action_advantages, human_preference)

    def get_objective_info(self, x):
        """This is used to get info from each agent regarding the state x"""
        state_values = []

        for i, agent in enumerate(self.agents):
            q_values = agent.get_actions(x)
            state_values.append(q_values)

        return state_values

    def store_memory(self, s, a, rewards, s_, d, info: dict):
        """Store experience to all agents"""
        for i in range(self.num_policies):
            self.agents[i].store_memory(s, a, rewards[i], s_, d)

    def get_loss_values(self) -> list[tuple[float, ...]]:
        """Get loss values for Q"""
        q_loss: list[tuple[float]] = []
        for i in range(self.num_policies):
            q_loss_part = self.agents[i].collect_loss_info()
            q_loss.append((q_loss_part))
        return q_loss

    def train(self) -> None:
        """Train all Q-networks"""
        for i in range(self.num_policies):
            self.agents[i].train()  # agents update their internal parameters as they go

    def update_params(self) -> None:
        """Update exploration rate"""
        self.exploration_strategy.update_parameters()

    def save(self, path: str) -> None:
        """Save all Q-networks"""
        for i in range(self.num_policies):
            self.agents[i].save_net(path + "Q" + str(i) + ".pt")

        self.exploration_strategy.save(path + "exploration_strategy.json")

    def load(self, path: str) -> None:
        """Load all Q-networks"""
        for i in range(self.num_policies):
            if os.path.exists(path + "Q" + str(i) + ".pt"):
                self.agents[i].load_net(path + "Q" + str(i) + ".pt")

        self.exploration_strategy.load(path + "exploration_strategy.json")

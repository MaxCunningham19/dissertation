import os

import numpy as np

from exploration_strategy import DecayEpsilonGreedy, ExplorationStrategy, Greedy
from agents.AbstractAgent import AbstractAgent
from ..dwn_agent import DWN
from ..dqn_agent import DQN


class DemocraticDWL(AbstractAgent):

    def __init__(
        self,
        input_shape,
        num_actions,
        num_policies,
        batch_size=1024,
        exploration_strategy: ExplorationStrategy = DecayEpsilonGreedy(epsilon=0.95, epsilon_decay=0.995, epsilon_min=0.1),
        memory_size=100000,
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

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_policies = num_policies  # The number of policies is the number of
        self.num_states = self.input_shape[0]
        self.init_learn_steps_count = 0
        self.init_learn_steps_num = init_learn_steps_num
        self.device = device
        self.seed = seed

        # Learning parameters for DQN agents
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy
        self.memory_size = memory_size

        # Construct Agents for each policy
        self.agents: list[tuple[DQN, DWN]] = []

        for i in range(self.num_policies):
            self.agents.append(
                (
                    DQN(
                        input_shape=self.input_shape,
                        num_actions=self.num_actions,
                        batch_size=self.batch_size,
                        tau=tau,
                        memory_size=self.memory_size,
                        learning_rate=self.learning_rate,
                        gamma=self.gamma,
                        per_epsilon=0.001,
                        beta_start=beta_start,
                        beta_inc=beta_inc,
                        exploration_strategy=Greedy(),  # each dqn has their own state
                        device=self.device,
                        hidlyr_nodes=hidlyr_nodes,
                    ),
                    DWN(
                        input_shape=self.input_shape,
                        batch_size=self.batch_size,
                        num_actions=self.num_actions,
                        tau=w_tau,
                        memory_size=self.memory_size,
                        learning_rate=self.learning_rate,
                        alpha=walpha,
                        gamma=self.gamma,
                        per_epsilon=0.001,
                        beta_start=beta_start,
                        beta_inc=beta_inc,
                        seed=self.seed,
                        device=self.device,
                        hidlyr_nodes=hidlyr_nodes,
                    ),
                )
            )

    def get_action(self, x) -> tuple[int, dict]:
        """Get the action nomination for the given state"""
        actions_values = np.array([0.0] * self.num_actions)
        for q_agent, w_agent in self.agents:
            q_values = np.array(q_agent.get_actions(x))
            w_value = w_agent.get_value(x)
            actions_values = actions_values + (q_values * w_value)

        action = self.exploration_strategy.get_action(actions_values, x)
        return action, {}

    def store_memory(self, s, a, rewards, s_, d, info: dict):
        for i in range(self.num_policies):
            q_agent, w_agent = self.agents[i]
            q_agent.store_memory(s, a, rewards[i], s_, d)
            w_agent.store_memory(s, a, rewards[i], s_, d)

    def get_loss_values(self) -> list[tuple[float, ...]]:
        """Get the loss values for Q and W"""
        losses: list[tuple[float, ...]] = []
        for q_agent, w_agent in self.agents:
            q_loss, w_loss = q_agent.collect_loss_info(), w_agent.collect_loss_info()
            losses.append((q_loss, w_loss))
        return losses

    def train(self) -> None:
        """Train Q and W networks"""
        for q_agent, w_agent in self.agents:
            q_agent.train()
            if self.init_learn_steps_count >= self.init_learn_steps_num:  # we start training W-network with delay
                w_agent.train()
        self.init_learn_steps_count += 1

    def update_params(self):
        """Update parameters for self and all agents"""
        for q_agent, w_agent in self.agents:
            q_agent.update_params()
            w_agent.update_params()
        self.exploration_strategy.update_parameters()

    def save(self, path):
        """Save trained Q-networks and W-networks to file"""
        for i, (q_agent, w_agent) in enumerate(self.agents):
            q_agent.save_net(path + "Q" + str(i) + ".pt")
            w_agent.save_net(path + "W" + str(i) + ".pt")

    def load(self, path):
        """Load the pre-trained Q-networks and W-networks from file, if they exist"""
        for i, (q_agent, w_agent) in enumerate(self.agents):
            if os.path.exists(path + "Q" + str(i) + ".pt"):
                q_agent.load_net(path + "Q" + str(i) + ".pt")
            if os.path.exists(path + "W" + str(i) + ".pt"):
                w_agent.load_net(path + "W" + str(i) + ".pt")

    def get_objective_info(self, x):
        """This is used to get info from each agent regarding the state x"""
        state_values = []

        for q_agent, w_agent in self.agents:
            q_values = np.array(q_agent.get_actions(x))
            w_values = w_agent.get_value(x)
            state_values.append(q_values * w_values)

        return state_values

    def get_all_info(self, x):
        """Get the weights for the given state"""
        weights = []
        q_values = []
        for q_agent, w_agent in self.agents:
            weights.append(w_agent.get_value(x))
            q_values.append(q_agent.get_actions(x))
        return weights, q_values

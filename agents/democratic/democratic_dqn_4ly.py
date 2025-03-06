import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

from exploration_strategy import ExplorationStrategy
from agents.AbstractAgent import AbstractAgent
from .dqn_agent import DQN


class DemocraticDQN(AbstractAgent):

    def __init__(
        self,
        input_shape,
        num_actions,
        num_policies,
        exploration_strategy: ExplorationStrategy,
        batch_size=1024,
        memory_size=10000,
        learning_rate=0.01,
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
        self.human_preference = human_preference
        if self.human_preference is None or len(human_preference) != self.num_policies:
            self.human_preference = [1.0 / self.num_policies] * self.num_policies

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
                    gamma=gamma,
                    per_epsilon=per_epsilon,
                    beta_start=beta_start,
                    beta_inc=beta_inc,
                    seed=seed,
                    device=device,
                    hidlyr_nodes=hidlyr_nodes,
                )
            )

    def get_status(self):
        """Returns the status of the agents learning params etc"""
        return ""

    def softmax(self, x):
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu()
            x = x.numpy()
        x = np.asarray(x).flatten()
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_action(self, x):
        action_advantages = np.array([0.0] * self.num_actions)

        for i, agent in enumerate(self.agents):
            q_values = agent.get_actions(x)
            scaled_q_values = self.softmax(q_values)
            preference_weighted_scaled_q_values = scaled_q_values * self.human_preference[i]
            action_advantages = action_advantages + preference_weighted_scaled_q_values

        return self.exploration_strategy.get_action(action_advantages, x), {}

    def get_actions(self, x):
        """Get every action from every agent"""
        action_advantages = np.array([0.0] * self.num_actions)

        for i, agent in enumerate(self.agents):
            q_values = agent.get_actions(x)
            scaled_q_values = self.softmax(q_values)
            preference_weighted_scaled_q_values = scaled_q_values * self.human_preference[i]
            action_advantages = action_advantages + preference_weighted_scaled_q_values

        return action_advantages

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
            print("Saving policy", i)
            self.agents[i].save_net(path + "Q" + str(i) + ".pt")

    def load(self, path: str) -> None:
        """Load all Q-networks"""
        for i in range(self.num_policies):
            print("Loading", i)
            self.agents[i].load_net(path + "Q" + str(i) + ".pt")

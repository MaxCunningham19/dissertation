import os

import numpy as np
import torch

from exploration_strategy.DecayEpsilonGreedy import DecayEpsilonGreedy
from exploration_strategy import ExplorationStrategy
from agents.AbstractAgent import AbstractAgent
from ..dwn_agent import DWN
from utils.utils import softmax


class DWL(AbstractAgent):

    def __init__(
        self,
        input_shape,
        num_actions,
        num_policies,
        exploration_strategy: ExplorationStrategy | None,
        w_exploration_strategy: ExplorationStrategy | None,
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
        q_learning=True,
        w_learning=True,
    ):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_policies = num_policies  # The number of policies is the number of
        self.num_states = self.input_shape[0]
        self.init_learn_steps_count = 0
        self.init_learn_steps_num = init_learn_steps_num
        self.device = device
        self.seed = seed
        self.q_learning = q_learning
        self.w_learning = w_learning

        # Learning parameters for DQN agents
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.w_exploration_strategy = w_exploration_strategy
        self.memory_size = memory_size

        # Construct Agents for each policy

        self.agents: list[DWN] = []

        for i in range(self.num_policies):
            self.agents.append(
                DWN(
                    input_shape=self.input_shape,
                    num_actions=self.num_actions,
                    batch_size=self.batch_size,
                    tau=tau,
                    memory_size=self.memory_size,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    per_epsilon=per_epsilon,
                    beta_start=beta_start,
                    beta_inc=beta_inc,
                    seed=self.seed,
                    exploration_strategy=exploration_strategy.copy(),
                    device=self.device,
                    hidlyr_nodes=hidlyr_nodes,
                    w_tau=w_tau,
                    w_alpha=w_alpha,
                )
            )

    def get_w_values(self, x, human_preference: np.ndarray | None = None) -> np.ndarray:
        """Get the weights for the given state"""
        if human_preference is None:
            human_preference = np.ones(self.num_policies)
        w_values = []
        for agent in self.agents:
            w_values.append(agent.get_w_value(x))
        softmax_w_values = softmax(w_values)
        softmax_w_values = softmax_w_values * human_preference
        # print(x, w_values, softmax_w_values)
        return softmax_w_values

    def get_action(self, x, human_preference: np.ndarray | None = None) -> tuple[int, dict]:
        """Get the action nomination for the given state"""
        w_values = self.get_w_values(x, human_preference)
        policy_sel = self.w_exploration_strategy.get_action(w_values, x)
        sel_action = self.agents[policy_sel].get_action(x)
        return sel_action, {"policy_sel": policy_sel, "w_values": w_values}

    def get_actions(self, x, human_preference: np.ndarray | None = None) -> np.ndarray:
        w_values = self.get_w_values(x, human_preference)
        policy_sel = self.w_exploration_strategy.get_action(w_values, x)
        # print(x, w_values, policy_sel)
        return self.agents[policy_sel].get_actions(x)

    def store_memory(self, s, a, rewards, s_, d, info: dict):
        for i, agent in enumerate(self.agents):
            if self.q_learning:
                agent.store_memory(s, a, rewards[i], s_, d)
            if self.w_learning and i != info["policy_sel"]:  # Do not store experience of the policy we selected
                agent.store_w_memory(s, a, rewards[i], s_, d)

    def get_loss_values(self) -> list[tuple[float, ...]]:
        """Get the loss values for Q and W"""
        losses: list[tuple[float, ...]] = []
        for agent in self.agents:
            losses.append(agent.collect_loss_info())
        return losses

    def train(self) -> None:
        """Train Q and W networks"""
        for agent in self.agents:
            if self.q_learning:
                agent.train()
            if self.w_learning and self.init_learn_steps_count >= self.init_learn_steps_num:  # we start training W-network with delay
                agent.train_w()
        self.init_learn_steps_count += 1

    def update_params(self):
        """Update parameters for self and all agents"""
        for agent in self.agents:
            agent.update_params()
        self.w_exploration_strategy.update_parameters()

    def save(self, path):
        """Save trained Q-networks and W-networks to file"""
        for i, agent in enumerate(self.agents):
            agent.save_net(path + "Q" + str(i) + ".pt")
            agent.save_w_net(path + "W" + str(i) + ".pt")
            agent.save_exploration_strategy(path + "exploration_strategy" + str(i) + ".json")
        self.w_exploration_strategy.save(path + "w_exploration_strategy.json")

    def load(self, path):
        """Load the pre-trained Q-networks and W-networks from file, if they exist"""
        for i, agent in enumerate(self.agents):
            if os.path.exists(path + "Q" + str(i) + ".pt"):
                agent.load_net(path + "Q" + str(i) + ".pt")
            if os.path.exists(path + "W" + str(i) + ".pt"):
                agent.load_w_net(path + "W" + str(i) + ".pt")
            if os.path.exists(path + "exploration_strategy" + str(i) + ".json"):
                agent.load_exploration_strategy(path + "exploration_strategy" + str(i) + ".json")
        if os.path.exists(path + "w_exploration_strategy.json") and self.w_exploration_strategy is not None:
            self.w_exploration_strategy.load(path + "w_exploration_strategy.json")

    def get_objective_info(self, x, human_preference: np.ndarray | None = None):
        """This is used to get info from each agent regarding the state x"""
        if human_preference is None:
            human_preference = np.ones(self.num_policies)

        state_values = []
        for i, agent in enumerate(self.agents):
            q_values = agent.get_actions(x)
            q_values = np.array(q_values) * human_preference[i]
            state_values.append(q_values)

        return state_values

    def get_all_info(self, x):
        """Get the weights for the given state"""
        weights = []
        q_values = []
        for agent in self.agents:
            weights.append(agent.get_w_value(x))
            q_values.append(agent.get_actions(x))
        return weights, q_values

    def softmax(self, x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x).flatten()
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def name(self) -> str:
        return "DWL"

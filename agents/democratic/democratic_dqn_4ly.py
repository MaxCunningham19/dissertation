import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

from ..dqn_agent_4ly import DQN


class DemocraticDQN(object):

    def __init__(
        self,
        input_shape,
        num_actions,
        num_policies,
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
        alpha=1.0,
        beta=1.5,
        evaporation_factor=0.9,
        pheromone_inc=0.8,
    ):

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_policies = num_policies
        self.num_states = self.input_shape[0]
        self.device = device
        self.human_preference = human_preference
        if self.human_preference is None or len(human_preference) != self.num_policies:
            self.human_preference = [1.0 / self.num_policies] * self.num_policies

        # Learning parameters for DQN agents
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory_size = memory_size
        self.tau = tau
        self.per_epsilon = per_epsilon
        self.alpha = alpha
        self.beta = beta
        self.evaporation_factor = evaporation_factor
        self.pheromone_inc = pheromone_inc
        self.pheromones = {}

        # Construct Agents for each policy
        self.agents: list[DQN] = []

        for i in range(self.num_policies):
            self.agents.append(
                DQN(
                    input_shape=self.input_shape,
                    num_actions=self.num_actions,
                    batch_size=self.batch_size,
                    tau=self.tau,
                    memory_size=self.memory_size,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    per_epsilon=self.per_epsilon,
                    seed=seed,
                    device=self.device,
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

    def get_action_nomination(self, x, printv=False, training=True):
        """Nominate an action"""
        action_advantages = np.array([0.0] * self.num_actions)

        for i, agent in enumerate(self.agents):
            q_values = agent.get_actions(x)
            scaled_q_values = self.softmax(q_values)
            preference_weighted_scaled_q_values = scaled_q_values * self.human_preference[i]
            action_advantages = action_advantages + preference_weighted_scaled_q_values
            if printv:
                print(i, q_values, scaled_q_values, self.human_preference[i], preference_weighted_scaled_q_values)

        if printv:
            print(f"Final: {action_advantages}")

        if training:
            return self.select_action(x, action_advantages)
        else:
            return np.argmax(action_advantages)

    def select_action(self, state, q_values):
        "returns the selected action based on current exploration strategy"
        state = tuple(state)
        if not state in self.pheromones:
            action = np.argmax(q_values)
            self.pheromones[state] = np.array([1.0] * self.num_actions)

        else:
            pheromones_state = np.array(self.pheromones[state])
            probabilities = self.softmax(((q_values) ** self.alpha) / ((pheromones_state) ** self.beta))
            action = np.random.choice(self.num_actions, p=probabilities)

        self.pheromones[state][action] = self.pheromones[state][action] + self.pheromone_inc

        return action

    def get_action(self, x, printv=False, training=True):
        """return an action"""
        return self.get_action_nomination(x, printv=printv, training=training)

    def get_actions(self, x):
        action_advantages = np.array([0.0] * self.num_actions)

        for i, agent in enumerate(self.agents):
            q_values = agent.get_actions(x)
            scaled_q_values = self.softmax(q_values)
            preference_weighted_scaled_q_values = scaled_q_values * self.human_preference[i]
            action_advantages = action_advantages + preference_weighted_scaled_q_values

        return action_advantages

    def get_agent_info(self, x):
        """This is used to get info from each agent regarding the state x"""
        state_values = []

        for i, agent in enumerate(self.agents):
            q_values = agent.get_actions(x)
            state_values.append(q_values)

        return state_values

    def store_transition(self, s, a, rewards, s_, d):
        """Store experience to all agents"""
        for i in range(self.num_policies):
            # print("storing for agent ", i, "-> (", s, a, rewards[i], s_, d, ")")
            self.agents[i].store_memory(s, a, rewards[i], s_, d)

    def store_memory(self, s, a, rewards, s_, d):
        """Store experience to all agents"""
        self.store_transition(s, a, rewards, s_, d)

    def get_loss_values(self):
        """Get loss values for Q"""
        q_loss = []
        for i in range(self.num_policies):
            q_loss_part = self.agents[i].collect_loss_info()
            q_loss.append(q_loss_part)
        return q_loss

    def train(self):
        """Train all Q-networks"""
        for i in range(self.num_policies):
            self.agents[i].train()  # agents update their internal parameters as they go
        self.update_params()

    def update_params(self):
        """Update exploration rate"""
        for state in self.pheromones:
            pheromones = self.pheromones[state]
            # print(state, ":", pheromones)
            for i in range(len(pheromones)):
                pheromones[i] = max(1.0, pheromones[i] * self.evaporation_factor)
            self.pheromones[state] = pheromones

    def save(self, path):
        """Save all Q-networks"""
        for i in range(self.num_policies):
            print("Saving policy", i)
            self.agents[i].save_net(path + "Q" + str(i) + ".pt")

    def load(self, path):
        """Load all Q-networks"""
        for i in range(self.num_policies):
            print("Loading", i)
            self.agents[i].load_net(path + "Q" + str(i) + ".pt")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

from .dueling_dqn_agent_4ly import DUELING_DQN


class DWL(object):

    def __init__(
        self,
        input_shape,
        num_actions,
        num_policies,
        batch_size=1024,
        epsilon=0.25,
        epsilon_decay=0.995,
        epsilon_min=0.1,
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
        self.device = device
        self.human_preference = human_preference
        if self.human_preference is None or len(human_preference) != self.num_policies:
            self.human_preference = [1.0 / self.num_policies] * self.num_policies

        self.action_state_count = {}

        # Learning parameters for DQN agents
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.tau = tau
        self.per_epsilon = per_epsilon

        # Construct Agents for each policy
        self.agents: list[DUELING_DQN] = []

        for i in range(self.num_policies):
            self.agents.append(
                DUELING_DQN(
                    input_shape=self.input_shape,
                    num_actions=self.num_actions,
                    batch_size=self.batch_size,
                    epsilon=self.epsilon,
                    epsilon_decay=self.epsilon_decay,
                    epsilon_min=self.epsilon_min,
                    tau=self.tau,
                    memory_size=self.memory_size,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    per_epsilon=self.per_epsilon,
                    beta_start=beta_start,
                    beta_inc=beta_inc,
                    seed=seed,
                    device=self.device,
                    hidlyr_nodes=hidlyr_nodes,
                )
            )

    def softmax(self, x):
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu()
            x = x.numpy()

        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_action_nomination(self, x, printv=False):
        """Nominate an action"""
        action_advantages = np.array([0.0] * self.num_actions)

        for i, agent in enumerate(self.agents):
            q_values, state_value, action_values = agent.get_actions(x)

            scaled_action_values = self.softmax(action_values)
            preference_weighted_scaled_action_values = scaled_action_values * self.human_preference[i]
            action_advantages = action_advantages + preference_weighted_scaled_action_values
            if printv:
                print(i, action_values, scaled_action_values, self.human_preference[i], preference_weighted_scaled_action_values)

        if printv:
            print(f"Final: {action_advantages}")
        if np.random.uniform() > self.epsilon:
            selected_action = np.argmax(action_advantages)
        else:
            selected_action = np.random.randint(0, self.num_actions)
        return selected_action

    def get_action(self, x, printv=False):
        """return an action"""
        action_nomination = self.get_action_nomination(x, printv=printv)
        if x in self.action_state_count:
            self.action_state_count[x][action_nomination] += 1

        return self.get_action_nomination(x, printv=printv)

    def get_agent_info(self, x):
        """This is used to get info from each agent regarding the state x"""
        state_values = []

        for i, agent in enumerate(self.agents):
            q_values, state_value, action_values = agent.get_actions(x)
            state_values.append((q_values, state_value, action_values))

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
            print("training ", i)
            self.agents[i].train()  # agents update their internal parameters as they go
        self.update_params()

    def update_params(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

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

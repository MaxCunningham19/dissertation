import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exploration_strategy import ExplorationStrategy
from exploration_strategy.utils import create_exploration_strategy, create_exploration_strategy_from_file
from .ReplayBuffer import ReplayBuffer


class DuelingQN(nn.Module):
    """
    Basic Model of a Q network
    """

    def __init__(self, state_number, hidlyr_nodes, action_number):
        super(DuelingQN, self).__init__()
        self.fc1 = nn.Linear(state_number, hidlyr_nodes)  # first conected layer
        self.fc2 = nn.Linear(hidlyr_nodes, hidlyr_nodes * 2)  # second conected layer
        self.fc3 = nn.Linear(hidlyr_nodes * 2, hidlyr_nodes * 4)  # second conected layer
        self.fc4 = nn.Linear(hidlyr_nodes * 4, hidlyr_nodes * 8)  # second conected layer
        self.values = nn.Linear(hidlyr_nodes * 8, 1)  # output layer for value function
        self.advantages = nn.Linear(hidlyr_nodes * 8, action_number)  # output layer

    def forward(self, state):
        x = F.relu(self.fc1(state))  # relu activation of fc1
        x = F.relu(self.fc2(x))  # relu activation of fc2
        x = F.relu(self.fc3(x))  # relu activation of fc2
        x = F.relu(self.fc4(x))  # relu activation of fc2
        values = self.values(x)  # calculate value function
        advantages = self.advantages(x)  # calculate action values
        x = values + (advantages - advantages.mean())  # combine value and action values
        return x


class DQN(object):
    def __init__(
        self,
        input_shape,
        num_actions,
        batch_size,
        tau,
        memory_size,
        learning_rate,
        gamma,
        per_epsilon,
        beta_start,
        beta_inc,
        seed,
        exploration_strategy: ExplorationStrategy | None,
        device,
        hidlyr_nodes,
    ):

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_states = self.input_shape[0]
        self.exploration_strategy = exploration_strategy
        self.random_seed = seed

        # Learning parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.memory_size = memory_size

        self.beta = beta_start
        self.beta_inc = beta_inc

        self.replayMemory = ReplayBuffer(self.num_actions, self.memory_size, self.batch_size, self.random_seed)

        self.per_epsilon = per_epsilon

        self.q_episode_loss = []

        self.policy_net = DuelingQN(self.num_states, hidlyr_nodes, self.num_actions)
        self.target_net = DuelingQN(self.num_states, hidlyr_nodes, self.num_actions)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        if device is not None:
            self.device = device
            self.policy_net.to(device)
            self.target_net.to(device)

    def get_actions(self, x):
        """Returns a list of actions and their associated q-values"""
        if type(x).__name__ == "ndarray":
            state = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        else:
            state = x
        a = self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)

        if isinstance(action_values, torch.Tensor):
            action_values = action_values.detach().cpu().numpy()
        action_values = np.array(action_values).flatten()

        return action_values.tolist()

    def get_action(self, x):
        """Nominate an action"""
        if type(x).__name__ == "ndarray":
            state = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        else:
            state = x
        a = self.policy_net.eval()
        action_values = self.get_actions(x)
        return self.exploration_strategy.get_action(action_values, state)

    def store_memory(self, state, action, reward, next_state, done):
        """Store memory in the PERBuffer"""
        self.replayMemory.add(state, action, reward, next_state, done)

    def train(self):
        """Train the model from the ReplayBuffe then update parameters"""
        if len(self.replayMemory) > self.batch_size:
            (states, actions, rewards, next_states, probabilities, experiences_idx, dones) = self.replayMemory.sample()
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            current_qs = self.policy_net(states).gather(1, actions)
            next_actions = self.policy_net(next_states).detach().max(1)[1].unsqueeze(1)
            max_next_qs = self.target_net(next_states).detach().gather(1, next_actions)
            target_qs = rewards + self.gamma * max_next_qs * (1.0 - (dones).float())

            is_weights = np.power(probabilities * self.batch_size, -self.beta)
            is_weights = torch.from_numpy(is_weights / is_weights.max()).float().to(self.device)
            loss = (target_qs - current_qs).pow(2) * is_weights
            loss = loss.mean()

            self.q_episode_loss.append(loss.detach().cpu().numpy())

            td_errors = (target_qs - current_qs).detach().cpu().numpy()
            self.replayMemory.update_priorities(experiences_idx, td_errors, self.per_epsilon)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.soft_update(self.policy_net, self.target_net, self.tau)

    def soft_update(self, originNet, targetNet, tau):
        """Update the target network towards the online network"""
        for target_param, local_param in zip(targetNet.parameters(), originNet.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_params(self):
        """Update model parameters"""
        self.beta = min(1.0, self.beta_inc * self.beta)
        self.exploration_strategy.update_parameters()

    def save_net(self, path):
        """Save the online network"""
        torch.save(self.policy_net.state_dict(), path)

    def load_net(self, path):
        """Load the network"""
        self.policy_net.load_state_dict(torch.load(path)), self.target_net.load_state_dict(torch.load(path))
        self.policy_net.eval(), self.target_net.eval()

    def collect_loss_info(self) -> float:
        """Get current loss value"""
        avg_q_loss = np.average(self.q_episode_loss)
        self.q_episode_loss = []
        return avg_q_loss

    def save_exploration_strategy(self, path):
        """Save the exploration strategy"""
        if self.exploration_strategy is not None:
            self.exploration_strategy.save(path)

    def load_exploration_strategy(self, path):
        """Load the exploration strategy"""
        if self.exploration_strategy is None:
            self.exploration_strategy = create_exploration_strategy_from_file(path)
        else:
            self.exploration_strategy.load(path)

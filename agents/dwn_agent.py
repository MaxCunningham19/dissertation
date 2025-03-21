import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ReplayBuffer import ReplayBuffer


class QN(nn.Module):
    """
    Basic Model of a Q network
    """

    def __init__(self, state_number, hidlyr_nodes, action_number):
        super(QN, self).__init__()
        self.fc1 = nn.Linear(state_number, hidlyr_nodes)  # first conected layer
        self.fc2 = nn.Linear(hidlyr_nodes, hidlyr_nodes * 2)  # second conected layer
        self.fc3 = nn.Linear(hidlyr_nodes * 2, hidlyr_nodes * 4)  # third conected layer
        self.fc4 = nn.Linear(hidlyr_nodes * 4, hidlyr_nodes * 8)  # fourth conected layer
        self.out = nn.Linear(hidlyr_nodes * 8, action_number)  # output layer

    def forward(self, state):
        x = F.relu(self.fc1(state))  # relu activation of fc1
        x = F.relu(self.fc2(x))  # relu activation of fc2
        x = F.relu(self.fc3(x))  # relu activation of fc3
        x = F.relu(self.fc4(x))  # relu activation of fc4
        x = self.out(x)  # calculate output
        return x


class DWN(object):

    def __init__(
        self,
        input_shape,
        num_actions,
        batch_size=1024,
        tau=0.001,
        memory_size=10000,
        learning_rate=0.01,
        alpha=0.001,
        gamma=0.9,
        per_epsilon=0.001,
        beta_start=0.4,
        beta_inc=1.002,
        seed=404,
        device=None,
        hidlyr_nodes=128,
    ):

        self.input_shape = input_shape
        self.num_states = self.input_shape[0]
        self.num_actions = num_actions
        self.random_seed = seed

        # Learning parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.memory_size = memory_size

        self.beta = beta_start
        self.beta_inc = beta_inc

        self.replayMemory = ReplayBuffer(
            action_size=self.num_actions, buffer_size=self.memory_size, batch_size=self.batch_size, seed=self.random_seed
        )
        self.per_epsilon = per_epsilon

        self.episode_loss = []

        self.wnetwork_local = QN(self.num_states, hidlyr_nodes, 1).to(device)
        self.wnetwork_target = QN(self.num_states, hidlyr_nodes, 1).to(device)
        self.optimizer = torch.optim.Adam(self.wnetwork_local.parameters(), lr=self.learning_rate)

        if device is not None:
            self.device = device
            self.wnetwork_local.to(device)
            self.wnetwork_target.to(device)

    def get_value(self, x):
        """Get the W-value for the given state"""
        if type(x).__name__ == "ndarray":
            state = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        else:
            state = x
        value = self.wnetwork_local.forward(state)
        value = value.detach()[0].cpu().data.numpy().flatten()
        value = value[0]
        return value

    def store_memory(self, s, a, r, s_, d):
        """Store the experience to the W-learning replay buffer"""
        self.replayMemory.add(s, a, r, s_, d)

    def train(self):
        """Train the W-network"""
        if len(self.replayMemory) > self.batch_size:
            (states, actions, rewards, next_states, probabilites, experiences_idx, dones) = self.replayMemory.sample()

            # Calculate the Q-values as in normal Q-learning
            current_qs = self.policy_net(states).gather(1, actions)
            next_actions = self.policy_net(next_states).detach().max(1)[1].unsqueeze(1)
            max_next_qs = self.target_net(next_states).detach().gather(1, next_actions)
            target_qs = rewards + self.gamma * max_next_qs * (1 - dones)

            # Calculate the W-values, as proposed with Eq. (3) in "W-learning Competition among selfish Q-learners"
            current_w = self.wnetwork_local(states).detach()
            target_w = (1 - self.alpha) * current_w + self.alpha * (current_qs - target_qs)

            is_weights = np.power(probabilites * self.batch_size, -self.beta)
            is_weights = torch.from_numpy(is_weights / np.max(is_weights)).float().to(self.device)
            loss = (target_w - current_w).pow(2) * is_weights
            loss = loss.mean()
            # To track the loss over episode
            self.episode_loss.append(loss.detach().cpu().numpy())

            td_errors = (target_qs - current_qs).detach().cpu().numpy()
            self.replayMemory.update_priorities(experiences_idx, td_errors, self.per_epsilon)

            self.wnetwork_local.zero_grad()
            loss.backward()
            self.optimizer.step()
            # ------------------- update target network ------------------- #
            self.soft_update(self.wnetwork_local, self.wnetwork_target, self.tau)

    def soft_update(self, originNet, targetNet, tau):
        """Update the target network towards the online network"""
        for target_param, local_param in zip(targetNet.parameters(), originNet.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_params(self):
        """Update parameters"""
        self.beta = min(1.0, self.beta_inc * self.beta)

    def save_net(self, path):
        """Save the W-network"""
        torch.save(self.wnetwork_local.state_dict(), path)

    def load_net(self, path):
        """Load the W-network"""
        self.wnetwork_local.load_state_dict(torch.load(path)), self.wnetwork_target.load_state_dict(torch.load(path))
        self.wnetwork_local.eval(), self.wnetwork_target.eval()

    def collect_loss_info(self):
        """Collect the loss information"""
        avg_loss = np.average(self.episode_loss)
        self.episode_loss = []
        return avg_loss

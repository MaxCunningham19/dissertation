import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from exploration_strategy import ExplorationStrategy
from ..ReplayBuffer import ReplayBuffer


class QN(nn.Module):
    """
    Basic Model of a Q network
    """

    def __init__(self, state_number, hidlyr_nodes, policy_number):
        super(QN, self).__init__()
        self.fc1 = nn.Linear(state_number + policy_number, hidlyr_nodes)  # first conected layer
        self.fc2 = nn.Linear(hidlyr_nodes, hidlyr_nodes * 2)  # second conected layer
        self.fc3 = nn.Linear(hidlyr_nodes * 2, hidlyr_nodes * 4)  # third conected layer
        self.fc4 = nn.Linear(hidlyr_nodes * 4, hidlyr_nodes * 8)  # fourth conected layer
        self.out = nn.Linear(hidlyr_nodes * 8, policy_number)  # output layer

    def forward(self, state):
        x = F.relu(self.fc1(state))  # relu activation of fc1
        x = F.relu(self.fc2(x))  # relu activation of fc2
        x = F.relu(self.fc3(x))  # relu activation of fc3
        x = F.relu(self.fc4(x))  # relu activation of fc4
        x = self.out(x)  # calculate output
        return x


class PreferenceNetwork(object):
    """This agent learns to output a preference weight for each policy to achieve a
    final episode reward that is a weighted sum of the rewards of the policies"""

    def __init__(
        self,
        input_shape,
        num_policies,
        batch_size=1024,
        tau=0.001,
        memory_size=10000,
        learning_rate=0.01,
        gamma=0.9,
        per_epsilon=0.001,
        beta_start=0.4,
        beta_inc=1.002,
        seed=404,
        exploration_strategy: ExplorationStrategy | None = None,
        device=None,
        hidlyr_nodes=128,
    ):

        self.input_shape = input_shape
        self.num_policies = num_policies
        self.num_states = self.input_shape[0]
        self.exploration_strategy = exploration_strategy
        self.random_seed = seed

        self.episode_buffer = []
        self.episode_reward = np.array([0.0] * self.num_policies)
        # Learning parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.memory_size = memory_size

        self.beta = beta_start
        self.beta_inc = beta_inc

        self.replayMemory = ReplayBuffer(self.num_policies, self.memory_size, self.batch_size, self.random_seed)

        self.per_epsilon = per_epsilon

        self.q_episode_loss = []

        self.policy_net = QN(self.num_states, hidlyr_nodes, self.num_policies)
        self.target_net = QN(self.num_states, hidlyr_nodes, self.num_policies)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        if device is not None:
            self.device = device
            self.policy_net.to(device)
            self.target_net.to(device)

    def get_weights(self, x):
        """Returns a list of policy weights"""
        if type(x).__name__ == "ndarray":
            state = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        else:
            state = x
        a = self.policy_net.eval()
        with torch.no_grad():
            policy_weights = self.policy_net(state)

        if isinstance(policy_weights, torch.Tensor):
            if policy_weights.is_cuda:
                policy_weights = policy_weights.cpu()
            policy_weights = policy_weights.numpy()
        policy_weights = policy_weights.flatten()

        return policy_weights.tolist()

    def store_memory(self, state, action, rewards, next_state, done):
        """Store memory in the PERBuffer
        if episode is done we store the reward as the sum of the final rewards else we store it in our buffer
        """
        if not done:
            self.episode_reward = self.episode_reward + np.array(rewards)
            self.episode_buffer.append((state, action, rewards, next_state, done))
        else:
            self.episode_rewards = self.episode_reward + np.array(rewards)
            self.episode_buffer.append((state, action, rewards, next_state, done))
            rewards_sum = np.array([0.0] * self.num_policies)
            for s, a, rs, s_, d in self.episode_buffer:
                rewards_sum = rewards_sum + np.array(rs)
                self.replayMemory.add(s, a, self.episode_rewards - rewards_sum, s_, d)
            self.episode_buffer = []
            self.episode_reward = np.array([0.0] * self.num_policies)

    def train(self):
        """Train the model from the ReplayBuffe then update parameters"""
        if len(self.replayMemory) > self.batch_size:
            (states, actions, epsiode_rewards, next_states, probabilities, experiences_idx, dones) = self.replayMemory.sample()

            current_qs = self.policy_net(states).gather(1, actions)
            next_actions = self.policy_net(next_states).detach().max(1)[1].unsqueeze(1)
            max_next_qs = self.target_net(next_states).detach().gather(1, next_actions)
            target_qs = rewards + self.gamma * max_next_qs * (1 - dones)

            is_weights = np.power(probabilities * len(self.replayMemory), -self.beta)
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

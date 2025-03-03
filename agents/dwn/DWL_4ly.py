from exploration_strategy import ExplorationStrategy, Greedy
from agents.AbstractAgent import AbstractAgent
from .dwn_agent_4ly import DWA


class DWL(object):

    def __init__(
        self,
        input_shape,
        num_actions,
        num_policies,
        w_learning=True,
        batch_size=1024,
        exploration_strategy: ExplorationStrategy = Greedy(),
        dwn_exploration_strategy: ExplorationStrategy = Greedy(),
        memory_size=10000,
        learning_rate=0.01,
        gamma=0.9,
        hidlyr_nodes=256,
        init_learn_steps_num=128,
        device=None,
    ):

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_policies = num_policies  # The number of policies is the number of
        self.num_states = self.input_shape[0]
        self.init_learn_steps_count = 0
        self.init_learn_steps_num = init_learn_steps_num
        self.device = device

        # Learning parameters for DQN agents
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_strategy = exploration_strategy
        self.memory_size = memory_size

        # Learning parameters for W learning net
        self.w_learning = w_learning
        self.dwn_exploration_strategy = dwn_exploration_strategy

        # Construct Agents for each policy
        self.agents: list[DWA] = []

        for i in range(self.num_policies):
            self.agents.append(
                DWA(
                    input_shape=self.input_shape,
                    num_actions=self.num_actions,
                    batch_size=self.batch_size,
                    exploration_strategy=self.dwn_exploration_strategy,
                    memory_size=self.memory_size,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    device=self.device,
                    hidlyr_nodes=hidlyr_nodes,
                    w_learning=self.w_learning,
                )
            )

    def get_action(self, x):
        """Get the action nomination for the given state"""
        nominated_actions = []
        w_values = []
        for agent in self.agents:
            nominated_actions.append(agent.get_action(x))
            w_values.append(agent.get_w_value(x))

        # Try different W-policies the same logic as exploration vs explotation
        policy_sel = self.exploration_strategy.get_action(w_values, x)
        sel_action = nominated_actions[policy_sel]
        return sel_action, policy_sel, nominated_actions

    def store_memory(self, s, a, rewards, s_, d, policy_sel):
        for i in range(self.num_policies):
            print(f"stored q memory in agent {i}")
            self.agents[i].store_memory(s, a, rewards[i], s_, d)
            if i != policy_sel:  # Do not store experience of the policy we selected
                print(f"stored w memory in agent {i}")
                self.agents[i].store_w_memory(s, a, rewards[i], s_, d)

    def get_loss_values(self):
        """Get the loss values for Q and W"""
        (q_loss, w_loss) = ([], [])
        for i in range(self.num_policies):
            q_loss_part, w_loss_part = self.agents[i].collect_loss_info()
            q_loss.append(q_loss_part), w_loss.append(w_loss_part)
        return q_loss, w_loss

    def train(self):
        """Train Q and W networks"""
        for i in range(self.num_policies):
            print("training pol :", i)
            self.agents[i].train()
            if self.init_learn_steps_count == self.init_learn_steps_num:  # we start training W-network with delay
                self.agents[i].learn_w()
        self.init_learn_steps_count += 1

    def update_params(self):
        """Update parameters for self and all agents"""
        for i in range(self.num_policies):
            self.agents[i].update_params()
        self.exploration_strategy.update_parameters()

    def save(self, path):
        """Save trained Q-networks and W-networks to file"""
        for i in range(self.num_policies):
            self.agents[i].save_net(path + "Q" + str(i) + ".pt")
            self.agents[i].save_w_net(path + "W" + str(i) + ".pt")

    def load(self, path):
        """Load the pre-trained Q-networks and W-networks from file"""
        for i in range(self.num_policies):
            print("Loading", i)
            self.agents[i].load_net(path + "Q" + str(i) + ".pt")
            self.agents[i].load_w_net(path + "W" + str(i) + ".pt")

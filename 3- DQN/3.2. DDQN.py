import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# ---------------------- Environment Setup ----------------------
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)


# ---------------------- Q-Network Definition ----------------------
class QNetwork(nn.Module):
    """Simple feedforward neural network for approximating Q-values"""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),   # First hidden layer
            nn.ReLU(),
            nn.Linear(64, 64),           # Second hidden layer
            nn.ReLU(),
            nn.Linear(64, action_size)   # Output layer (Q-values per action)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------- Replay Memory ----------------------
class ReplayBuffer:
    """Stores past experiences for stable training"""
    def __init__(self, capacity=10000):
        self.memory = []
        self.capacity = capacity

    def push(self, experience):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ---------------------- DDQN Algorithm ----------------------
class DDQN:
    def __init__(self, num_episodes, epsilon, discount, alpha, batch_size=64, update_target_every=1000):
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.discount = discount
        self.alpha = alpha
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        # FrozenLake has discrete states -> represent as one-hot vectors
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n

        # Networks: online (for selecting actions) and target (for stability)
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Initialize same weights

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

        # Replay buffer
        self.memory = ReplayBuffer()

    def one_hot(self, state):
        """Convert integer state to one-hot tensor"""
        v = np.zeros(self.state_size)
        v[state] = 1
        return torch.FloatTensor(v).unsqueeze(0)

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            with torch.no_grad():
                q_values = self.q_network(self.one_hot(state))
            return q_values.argmax().item()

    def train_step(self):
        """Perform a single training step using a batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat([self.one_hot(s) for s in states])
        next_states = torch.cat([self.one_hot(s) for s in next_states])
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q estimates
        q_values = self.q_network(states).gather(1, actions)

        # Next Q values using DDQN method:
        # 1. Choose best action using online Q-network
        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        # 2. Evaluate with target network
        next_q_values = self.target_network(next_states).gather(1, next_actions).detach()

        # Bellman target
        target_q = rewards + self.discount * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def MainAlgorithm(self):
        step_count = 0

        for episode in range(self.num_episodes):
            state = env.reset()[0]
            done = False

            while not done:
                # Choose action (ε-greedy)
                action = self.select_action(state)

                # Take step in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Store experience
                self.memory.push((state, action, reward, next_state, done))

                # Train network
                self.train_step()

                # Move to next state
                state = next_state
                step_count += 1

                # Periodically update target network
                if step_count % self.update_target_every == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

        # Derive final policy
        policy = []
        for s in range(self.state_size):
            with torch.no_grad():
                q_values = self.q_network(self.one_hot(s))
                policy.append(q_values.argmax().item())
        return np.array(policy)


# ---------------------- Run DDQN ----------------------
n_episodes = 5000
epsilon = 0.1
discount = 0.99
alpha = 0.001

ddqn = DDQN(n_episodes, epsilon, discount, alpha)

policy = ddqn.MainAlgorithm()

env.close()
print("------------- Testing -------------")
print(policy)


# ---------------------- Testing Learned Policy ----------------------
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
state = env.reset()[0]
done = False

while not done:
    action = policy[state]
    next_state, reward, terminated, truncated, _ = env.step(action)
    env.render()
    done = terminated or truncated
    state = next_state

env.close()

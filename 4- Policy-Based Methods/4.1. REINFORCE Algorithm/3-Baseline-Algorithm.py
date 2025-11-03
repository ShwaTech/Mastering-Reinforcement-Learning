"""
=========================================
REINFORCE with Baseline — Continuous Action Space
=========================================

Implements REINFORCE with a learned value baseline (advantage estimation).
The baseline reduces variance by subtracting V(s) from returns.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Environment and setup
# -----------------------------
env = gym.make("Pendulum-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low, action_high = env.action_space.low[0], env.action_space.high[0]


# -----------------------------
# Hyperparameters
# -----------------------------
GAMMA = 0.99
LR_POLICY = 3e-4
LR_VALUE = 1e-3
NUM_EPISODES = 600
MAX_STEPS = 200
ENTROPY_BETA = 1e-3
LOG_STD_MIN, LOG_STD_MAX = -20, 2
NORMALIZE_ADV = True
torch.manual_seed(1)
np.random.seed(1)


# -----------------------------
# Networks
# -----------------------------
class GaussianPolicy(nn.Module):
    """Gaussian policy network for continuous control."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.mu = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
    
    
    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mu, std)


class ValueNetwork(nn.Module):
    """Baseline network predicting V(s)."""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.net(state).squeeze(-1)


# -----------------------------
# Utilities
# -----------------------------
def compute_reward_to_go(rewards, gamma):
    """Compute discounted returns for each timestep."""
    rtg = np.zeros_like(rewards, dtype=np.float32)
    
    running = 0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        rtg[t] = running
    
    return rtg


# -----------------------------
# Training Loop
# -----------------------------
policy = GaussianPolicy(state_dim, action_dim)
value_net = ValueNetwork(state_dim)
opt_policy = optim.Adam(policy.parameters(), lr=LR_POLICY)
opt_value = optim.Adam(value_net.parameters(), lr=LR_VALUE)


for ep in range(1, NUM_EPISODES + 1):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32).flatten()
    rewards, log_probs, entropies, states = [], [], [], []
    total_reward = 0
    
    for step in range(MAX_STEPS):
        s_tensor = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
        dist = policy(s_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        entropy = dist.entropy().sum()
        
        action_clipped = np.clip(action.detach().numpy(), action_low, action_high)
        next_state, reward, terminated, truncated, _ = env.step(action_clipped)
        reward = float(reward)
        next_state = np.array(next_state, dtype=np.float32).flatten()
        done = terminated or truncated
        
        states.append(state)
        rewards.append(reward)
        log_probs.append(log_prob)
        entropies.append(entropy)
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    # Compute returns (Reward-to-Go)
    returns = torch.tensor(compute_reward_to_go(rewards, GAMMA))
    states_tensor = torch.tensor(np.vstack(states), dtype=torch.float32)
    
    # Compute baseline (V(s))
    values = value_net(states_tensor).detach()
    advantages = returns - values
    
    if NORMALIZE_ADV:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)
    
    # Policy loss (use advantage)
    policy_loss = -torch.sum(log_probs * advantages) - ENTROPY_BETA * entropies.sum()
    
    opt_policy.zero_grad()
    policy_loss.backward()
    opt_policy.step()
    
    # Baseline (value) loss — regression to returns
    value_pred = value_net(states_tensor)
    value_loss = nn.MSELoss()(value_pred, returns)
    opt_value.zero_grad()
    value_loss.backward()
    opt_value.step()
    
    if ep % 10 == 0:
        print(f"Episode {ep:3d} | Return: {total_reward:7.2f} | PolicyLoss: {policy_loss.item():.4f} | ValueLoss: {value_loss.item():.4f}")


env.close()
print("✅ Training (Baseline) complete!")

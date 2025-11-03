"""
=========================================
REINFORCE (Reward-to-Go) — Continuous Action Space
=========================================

Implements vanilla REINFORCE using a Gaussian policy for continuous actions
(e.g., Pendulum-v1) with reward-to-go returns and no baseline.
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
env = gym.make("Pendulum-v1", render_mode="human")  # Use "human" to visualize
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low, action_high = env.action_space.low[0], env.action_space.high[0]


# -----------------------------
# Hyperparameters
# -----------------------------
GAMMA = 0.99
LR = 3e-4
NUM_EPISODES = 600
MAX_STEPS = 200
LOG_STD_MIN, LOG_STD_MAX = -20, 2
ENTROPY_BETA = 1e-3
NORMALIZE_RETURNS = True
torch.manual_seed(1)
np.random.seed(1)


# -----------------------------
# Gaussian Policy Network
# -----------------------------
class GaussianPolicy(nn.Module):
    """
    A Gaussian policy that outputs μ and σ for each state,
    then samples continuous actions accordingly.
    """
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
        std = log_std.exp()
        return torch.distributions.Normal(mu, std)


# -----------------------------
# Helper: Compute Reward-to-Go
# -----------------------------
def compute_reward_to_go(rewards, gamma):
    """
    For each timestep t, compute discounted sum of future rewards:
        G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
    """
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
optimizer = optim.Adam(policy.parameters(), lr=LR)


for ep in range(1, NUM_EPISODES + 1):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32).flatten()
    rewards, log_probs, entropies = [], [], []
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
        
        rewards.append(reward)
        log_probs.append(log_prob)
        entropies.append(entropy)
        total_reward += reward
        
        state = next_state
        if done:
            break
    
    # Compute reward-to-go
    returns = torch.tensor(compute_reward_to_go(rewards, GAMMA))
    
    # Normalize for stability
    if NORMALIZE_RETURNS:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)
    
    # Policy gradient loss
    policy_loss = -torch.sum(log_probs * returns) - ENTROPY_BETA * entropies.sum()
    
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    if ep % 10 == 0:
        print(f"Episode {ep:3d} | Return: {total_reward:7.2f} | PolicyLoss: {policy_loss.item():.4f}")


env.close()
print("✅ Training (Reward-to-Go) complete!")

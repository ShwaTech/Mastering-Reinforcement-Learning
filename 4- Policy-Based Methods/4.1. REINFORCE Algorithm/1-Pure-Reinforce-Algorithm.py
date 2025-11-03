"""
REINFORCE (Vanilla Policy Gradient) with Gaussian (Normal) Policy
Environment: Pendulum-v1  (continuous action space)

This is the continuous-action version of REINFORCE.
The policy outputs (mean, std) of a Normal distribution.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------
# 1. Reproducibility
# -------------------------
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------
# 2. Hyperparameters
# -------------------------
ENV_NAME = "Pendulum-v1"     # continuous actions in [-2, 2]
GAMMA = 0.99                 # discount factor
LR = 3e-4                    # learning rate
NUM_EPISODES = 600           # total episodes
MAX_STEPS = 200              # max steps per episode (Pendulum default)
NORMALIZE_RETURNS = True     # stabilize training
ENTROPY_BETA = 1e-3          # small entropy bonus (encourage exploration)
LOG_STD_MIN = -20            # clamp minimum log std for numerical stability
LOG_STD_MAX = 2              # clamp maximum log std


# -------------------------
# 3. Environment setup
# -------------------------
env = gym.make(ENV_NAME, render_mode="human")
env.reset(seed=SEED)

state_dim = env.observation_space.shape[0]   # Pendulum: 3-dimensional state
action_dim = env.action_space.shape[0]       # Pendulum: 1 continuous action
action_low = float(env.action_space.low[0])
action_high = float(env.action_space.high[0])

print(f"State dim: {state_dim}, Action dim: {action_dim}, Action range: [{action_low}, {action_high}]")

# -------------------------
# 4. Gaussian Policy Network
# -------------------------
class GaussianPolicy(nn.Module):
    """
    Outputs a Normal distribution for continuous actions:
    μ(s), σ(s)  -> action sampled from  N(μ, σ)
    """
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super(GaussianPolicy, self).__init__()
        layers = []
        last_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.shared = nn.Sequential(*layers)
        
        # Output layers for mean (μ) and log standard deviation (log σ)
        self.mu_layer = nn.Linear(last_dim, action_dim)
        self.log_std_layer = nn.Linear(last_dim, action_dim)
    
    
    def forward(self, state):
        """
        Forward pass:
        Input: state (batch, state_dim)
        Output: distribution (torch.distributions.Normal)
        """
        x = self.shared(state)
        mu = self.mu_layer(x)
        
        # Clamp log std to prevent numerical instability
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # Create a Normal distribution parameterized by (μ, σ)
        dist = torch.distributions.Normal(mu, std)
        return dist


# Instantiate policy and optimizer
policy = GaussianPolicy(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=LR)


# -------------------------
# 5. Compute discounted returns
# -------------------------
def compute_discounted_returns(rewards, gamma):
    """Compute discounted sum of future rewards (returns) G_t."""
    returns = np.zeros_like(rewards, dtype=np.float32)
    
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    
    return returns


# -------------------------
# 6. Training Loop (REINFORCE)
# -------------------------
for episode in range(1, NUM_EPISODES + 1):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    
    log_probs = []
    rewards = []
    entropies = []
    
    total_reward = 0.0
    
    for step in range(MAX_STEPS):
        # Convert state to torch tensor
        state_tensor = torch.from_numpy(state).unsqueeze(0)
        
        # Get Gaussian distribution from policy
        dist = policy(state_tensor)
        
        # Sample an action from this distribution
        action = dist.sample()
        
        # Compute log probability and entropy
        log_prob = dist.log_prob(action).sum(dim=-1)    # sum if multi-dimensional
        entropy = dist.entropy().sum(dim=-1)
        
        # Rescale action to environment's action range
        scaled_action = torch.clamp(action, action_low, action_high)
        
        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(scaled_action.detach().numpy().flatten())
        done = terminated or truncated
        
        # Store experience
        log_probs.append(log_prob)
        entropies.append(entropy)
        rewards.append(reward)
        
        total_reward += reward
        state = np.array(next_state, dtype=np.float32)
        
        if done:
            break
    
    # Compute discounted returns G_t
    returns = compute_discounted_returns(rewards, GAMMA)
    
    # Normalize returns (helps reduce variance)
    if NORMALIZE_RETURNS:
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)
    
    # Compute policy gradient loss:
    # J = E[ log π(a_t|s_t) * G_t ]
    # We minimize negative objective for gradient ascent.
    policy_loss = - (log_probs * returns).sum()
    
    # Add small entropy bonus to encourage exploration
    policy_loss -= ENTROPY_BETA * entropies.sum()
    
    # Gradient update
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    # Logging
    avg_return = np.mean(rewards)
    print(f"Episode {episode:3d} | Return: {total_reward:8.2f} | Steps: {len(rewards):3d} | Loss: {policy_loss.item():.4f}")


env.close()
print("\nTraining completed ✅")

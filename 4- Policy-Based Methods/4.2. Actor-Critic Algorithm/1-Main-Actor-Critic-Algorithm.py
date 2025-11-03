"""
Vanilla Actor-Critic (REINFORCE-with-learned-baseline style) — corrected, robust implementation.

Environment: Pendulum-v1 (continuous action space)

What this implements:
- Actor: Gaussian policy (state-dependent mean; learnable log_std)
- Critic: Value function V(s) approximator (scalar output)
- Trajectory: full-episode Monte-Carlo returns G_t
- Advantage: A_t = G_t - V(s_t)  (learned baseline)
- Updates after each episode:
    * actor_loss = - E_t [ log π(a_t|s_t) * A_t ]  (we minimize negative objective)
    * critic_loss = E_t [ (G_t - V(s_t))^2 ]
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time


# -----------------------------
# Configuration / hyperparams
# -----------------------------
ENV_NAME = "Pendulum-v1"
SEED = 1
NUM_EPISODES = 800           # number of training episodes
MAX_STEPS = 200              # env limit per episode
GAMMA = 0.99                 # discount factor for returns
LR_ACTOR = 3e-4              # learning rate for actor
LR_CRITIC = 1e-3             # learning rate for critic
HIDDEN_UNITS = 128          # hidden layer size
LOG_STD_MIN = -20.0         # clamp for log std
LOG_STD_MAX = 2.0
NORMALIZE_RETURNS = True    # normalize returns per episode (recommended)
ENTROPY_COEF = 1e-3         # small entropy bonus to encourage exploration
DEVICE = torch.device("cpu")  # keep on CPU by default; change if GPU available


# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------------
# Environment setup
# -----------------------------
# If you want to visualize, pass render_mode="human" and ensure a display is available.
# For headless runs set render_mode=None (default).
env = gym.make(ENV_NAME, render_mode="human")
env.reset(seed=SEED)

# Get state and action dimensions
state_dim = env.observation_space.shape[0]   # Pendulum: 3
action_dim = env.action_space.shape[0]       # Pendulum: 1
action_low = float(env.action_space.low[0])
action_high = float(env.action_space.high[0])


# scaling from unconstrained tanh[-1,1] to env action range if you choose to use tanh
# here we will sample raw Gaussian and clip to bounds (simpler for REINFORCE baseline).
action_scale = (action_high - action_low) / 2.0
action_bias = (action_high + action_low) / 2.0


# Print environment info
print(f"Env: {ENV_NAME} | state_dim={state_dim} | action_dim={action_dim} | action_range=[{action_low}, {action_high}]")
print("Vanilla Actor-Critic (Monte-Carlo returns, learned baseline).")


# -----------------------------
# Network definitions
# -----------------------------
class Actor(nn.Module):
    """
    Actor: outputs a Gaussian distribution for the action.
    We use a state-dependent mean and a learnable log_std vector (per-dim).
    Outputs a torch.distributions.Normal(mean, std) object given state(s).
    """
    def __init__(self, state_dim, action_dim, hidden_units=HIDDEN_UNITS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU()
        )
        # mean head
        self.mean = nn.Linear(hidden_units, action_dim)
        # log_std is a parameter vector (one value per action dim)
        # using Parameter gives a single learnable vector (global std), which is stable
        self.log_std_param = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
    
    def forward(self, state_tensor):
        """
        state_tensor shape: (batch, state_dim)
        returns: Normal(loc=mean, scale=std) distribution (torch.distributions.Normal)
        """
        x = self.net(state_tensor)
        mean = self.mean(x)                                # shape: (batch, action_dim)
        # clip log_std param for numerical stability
        log_std = torch.clamp(self.log_std_param, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)                           # shape: (action_dim,)
        # Expand std to match batch: (batch, action_dim)
        std = std.unsqueeze(0).expand_as(mean)
        return Normal(loc=mean, scale=std)


class Critic(nn.Module):
    """
    Critic: outputs a scalar V(s) estimate.
    """
    def __init__(self, state_dim, hidden_units=HIDDEN_UNITS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )
    
    def forward(self, state_tensor):
        """
        state_tensor shape: (batch, state_dim)
        returns: value tensor shape (batch, 1) -> we will squeeze to (batch,)
        """
        return self.net(state_tensor).squeeze(-1)


# -----------------------------
# Utilities
# -----------------------------
def preprocess_state(state):
    """
    Ensure state is a 1-D numpy float32 array of length state_dim.
    This guards against shapes like (3,1) or nested lists from env wrappers.
    """
    arr = np.asarray(state, dtype=np.float32).flatten()
    if arr.shape[0] != state_dim:
        raise ValueError(f"Unexpected state shape after flatten: {arr.shape}")
    return arr


def compute_episode_returns(rewards, gamma):
    """
    Compute Monte-Carlo returns G_t for an episode (full returns).
    Input: rewards list [r0, r1, ..., r_{T-1}]
    Output: numpy array returns [G0, G1, ..., G_{T-1}] where
        G_t = sum_{k=t}^{T-1} gamma^{k-t} * r_k
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


# -----------------------------
# Instantiate actor, critic, optimizers
# -----------------------------
actor = Actor(state_dim, action_dim).to(DEVICE)
critic = Critic(state_dim).to(DEVICE)

actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)


# -----------------------------
# Training loop
# -----------------------------
start_time = time.time()
episode_returns = []
for episode in range(1, NUM_EPISODES + 1):
    state, _ = env.reset()
    state = preprocess_state(state)
    
    log_probs = []    # list of torch scalars (log probability of action per step)
    values = []       # list of torch scalars V(s_t)
    rewards = []      # list of floats
    entropies = []    # optional entropy terms for bonus
    
    ep_return = 0.0
    
    # Collect full episode (Monte Carlo)
    for step in range(MAX_STEPS):
        # prepare state tensor shape (1, state_dim)
        s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)  # shape (1, state_dim)
        
        # Get policy distribution and value estimate
        dist = actor(s_tensor)                  # torch.distributions.Normal
        value = critic(s_tensor)                # shape (1,)
        
        # Sample action (stochastic policy)
        action_tensor = dist.sample()           # shape (1, action_dim)
        # log_prob for sampled action, summed across action dims -> shape (1,)
        log_prob = dist.log_prob(action_tensor).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)    # entropy approx for exploration
        
        # Convert action to numpy and ensure correct bounds and dtype -> feed env
        # We clip to the environment action range to avoid out-of-bound actions.
        action_np = action_tensor.detach().cpu().numpy().reshape(action_dim,)
        action_np_clipped = np.clip(action_np, action_low, action_high).astype(np.float32)
        
        # Step environment
        next_state, reward, terminated, truncated, _ = env.step(action_np_clipped)
        done = bool(terminated or truncated)
        
        # Cast reward to python float to avoid Gym warning about numpy types
        reward = float(reward)
        
        # Save trajectory elements
        log_probs.append(log_prob.squeeze(0))   # tensor scalar
        values.append(value.squeeze(0))         # tensor scalar
        entropies.append(entropy.squeeze(0))
        rewards.append(reward)
        ep_return += reward
        
        # Prepare next state
        state = preprocess_state(next_state)
        
        if done:
            break
    
    T = len(rewards)
    if T == 0:
        # Safety guard: no timesteps in episode (should not happen in Pendulum)
        continue
    
    # Convert returns: Monte Carlo full-episode returns
    returns_np = compute_episode_returns(rewards, GAMMA)
    
    # Optionally normalize returns (recommended to reduce scale issues)
    if NORMALIZE_RETURNS:
        returns_np = (returns_np - np.mean(returns_np)) / (np.std(returns_np) + 1e-8)
    
    # Convert lists to tensors (shape (T,))
    returns = torch.tensor(returns_np, dtype=torch.float32, device=DEVICE)     # shape (T,)
    log_probs_tensor = torch.stack(log_probs).to(DEVICE)                       # shape (T,)
    values_tensor = torch.stack(values).to(DEVICE)                             # shape (T,)
    entropies_tensor = torch.stack(entropies).to(DEVICE)                       # shape (T,)
    
    # Advantage: A_t = G_t - V(s_t)
    advantages = returns - values_tensor.detach()   # detach critic predictions from actor update
    
    # Actor loss: - E_t[ log π(a_t|s_t) * A_t ]  (we minimize negative)
    # We'll add a small entropy bonus to encourage exploration (optional)
    actor_loss = - (log_probs_tensor * advantages).mean() - ENTROPY_COEF * entropies_tensor.mean()
    
    # Critic loss: MSE between predicted V(s) and MC returns
    critic_loss = nn.functional.mse_loss(values_tensor, returns)
    
    # --- Update actor ---
    actor_optimizer.zero_grad()
    actor_loss.backward()
    # optional grad clipping for stability
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
    actor_optimizer.step()
    
    # --- Update critic ---
    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
    critic_optimizer.step()
    
    # Logging / bookkeeping
    episode_returns.append(ep_return)
    if len(episode_returns) > 100:
        episode_returns.pop(0)
    avg100 = float(np.mean(episode_returns))
    
    if episode % 10 == 0 or episode == 1:
        elapsed = time.time() - start_time
        print(f"Ep {episode:4d} | Return: {ep_return:8.2f} | Steps: {T:3d} | "
              f"ActorLoss: {actor_loss.item():.4f} | CriticLoss: {critic_loss.item():.4f} | Avg100: {avg100:6.2f} | Time: {elapsed:.1f}s")


# Close env cleanly
env.close()
print("Training finished ✅")


# -------------------------------------------------
# Deterministic evaluation (use policy mean)
# -------------------------------------------------
def evaluate_policy(actor_model, env_name, n_episodes=5, render=False):
    eval_env = gym.make(env_name, render_mode="human" if render else None)
    rewards = []
    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        state = preprocess_state(state)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < MAX_STEPS:
            s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                # deterministic action = mean
                dist = actor_model(s_tensor)
                mean_action = dist.mean.cpu().numpy().reshape(action_dim,)
                action_np_clipped = np.clip(mean_action, action_low, action_high).astype(np.float32)
            next_state, reward, terminated, truncated, _ = eval_env.step(action_np_clipped)
            reward = float(reward)
            total += reward
            state = preprocess_state(next_state)
            done = bool(terminated or truncated)
            steps += 1
            if render:
                eval_env.render()
        rewards.append(total)
    eval_env.close()
    return np.mean(rewards), np.std(rewards)


mean_r, std_r = evaluate_policy(actor, ENV_NAME, n_episodes=5, render=False)
print(f"Evaluation (deterministic mean) over 5 episodes: {mean_r:.2f} ± {std_r:.2f}")

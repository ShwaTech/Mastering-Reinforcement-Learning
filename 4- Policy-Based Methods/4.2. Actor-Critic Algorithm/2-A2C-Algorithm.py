"""
A2C (Synchronous Advantage Actor-Critic) for continuous actions (Pendulum-v1)

- Single-process, synchronous A2C that collects n-step rollouts and updates actor & critic.
- Shared backbone network with two heads (actor + critic).
- Actor outputs Gaussian mean and log_std; we sample with reparameterization and tanh-squash.
- Log-prob correction for tanh transform is included (change-of-variables).
- Critic trained to predict n-step bootstrapped returns (TD(n) target).
- Advantage used for policy update: A = R^{(n)} + γ^n V(s_{t+n}) - V(s_t)

Important references / equations used:
    - n-step return (bootstrapped):
        R_t^{(n)} = sum_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n}) * (1 - done_{t+n})
    - Advantage:
        A_t = R_t^{(n)} - V(s_t)
    - Actor loss (we minimize negative objective):
        L_actor = - E_t [ log π(a_t|s_t) * A_t ] - β * H[π(.|s_t)]
    - Critic loss:
        L_critic = E_t [ (R_t^{(n)} - V(s_t))^2 ]
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import warnings


# -------------------------
# Config / Hyperparameters
# -------------------------
ENV_NAME = "Pendulum-v1"
SEED = 1
NUM_UPDATES = 1000          # number of A2C update iterations (not episodes)
N_STEPS = 5                 # n-step return length
GAMMA = 0.99
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
HIDDEN = 128
ENTROPY_COEF = 1e-3
VALUE_COEF = 0.5            # weight for critic loss
MAX_EPISODE_LEN = 200
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
GRAD_CLIP = 0.5
NORMALIZE_ADV = True        # normalize advantages per update (optional)
DEVICE = torch.device("cpu")  # change to "cuda" if GPU available


# reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------
# Environment
# -------------------------
env = gym.make(ENV_NAME, render_mode="human")  # change to "human" if you want visualization
env.reset(seed=SEED)


# Get state and action dimensions
state_dim = env.observation_space.shape[0]   # Pendulum: 3
action_dim = env.action_space.shape[0]       # Pendulum: 1
action_low = float(env.action_space.low[0])
action_high = float(env.action_space.high[0])


# mapping for tanh-squash -> env range: action_env = tanh(pre) * scale + bias
action_scale = (action_high - action_low) / 2.0
action_bias = (action_high + action_low) / 2.0

print(f"A2C continuous (Pendulum) | n_steps={N_STEPS} | updates={NUM_UPDATES}")


# -------------------------
# Shared network with actor & critic heads
# -------------------------
class SharedActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN):
        super().__init__()
        # shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # actor head: outputs mean (dim = action_dim)
        self.mean_head = nn.Linear(hidden, action_dim)
        # learnable log_std parameter (global per-dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        # critic head: outputs scalar V(s)
        self.value_head = nn.Linear(hidden, 1)
    
    
    def forward(self, state):
        """
        state: tensor shape (batch, state_dim)
        returns:
            mean (batch, action_dim), log_std (action_dim), value (batch,)
        """
        x = self.backbone(state)
        mean = self.mean_head(x)
        # clamp log_std for stability then expand to batch
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)                  # shape (action_dim,)
        value = self.value_head(x).squeeze(-1)    # shape (batch,)
        return mean, std, value
    
    
    def sample_action(self, state):
        """
        Sample an action using reparameterization and tanh-squash.
        Returns:
            action_env (numpy array clipped to env bounds),
            log_prob (torch tensor shape (batch,)),  -- corrected for tanh
            entropy (torch tensor shape (batch,))
        """
        mean, std, _ = self.forward(state)  # mean (batch, A), std (A,), value ignored here
        # reparameterize: sample eps ~ N(0,1)
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std.unsqueeze(0)    # shape (batch, A)
        tanh_action = torch.tanh(pre_tanh)          # in (-1,1)
        # scale to env range
        action_env = tanh_action * action_scale + action_bias
        
        # compute log_prob correction:
        # log_prob(pre_tanh) under Normal(mean, std)
        normal = torch.distributions.Normal(mean, std.unsqueeze(0))
        logp_pre = normal.log_prob(pre_tanh).sum(dim=-1)  # sum dims -> (batch,)
        # correction term: sum log(1 - tanh^2(pre_tanh) + eps)
        # (derivative of tanh is (1 - tanh^2))
        eps_val = 1e-6
        log_det = torch.log(1 - tanh_action.pow(2) + eps_val).sum(dim=-1)
        # corrected log_prob:
        log_prob = logp_pre - log_det
        
        # entropy (approx) of policy (use normal entropy) summed across action dims
        entropy = normal.entropy().sum(dim=-1)
        
        return action_env.detach().cpu().numpy(), log_prob, entropy, tanh_action, pre_tanh


# -------------------------
# Utilities
# -------------------------
def preprocess_state(state):
    """
    Ensure state is a 1-D numpy float32 array (shape (state_dim,))
    """
    s = np.asarray(state, dtype=np.float32).flatten()
    if s.shape[0] != state_dim:
        raise ValueError(f"State has unexpected shape after flatten: {s.shape}")
    return s

def compute_nstep_returns(rewards, last_value, gamma, dones):
    """
    Compute n-step bootstrapped returns for trajectory of length T:
      R_t^{(n)} = r_t + gamma*r_{t+1} + ... + gamma^{T-1-t}*r_{T-1} + gamma^{T-t} * V(s_T) * (1 - done_T)
    Input:
        rewards: list/np.array length T
        last_value: scalar V(s_T) (torch or float)
        dones: list/booleans length T indicating whether episode ended at that step
    Output:
        returns: numpy array length T with bootstrapped returns for each t
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    R = float(last_value)  # bootstrap value for time T (if last step not terminal)
    
    for t in reversed(range(T)):
        R = rewards[t] + gamma * R * (0.0 if dones[t] else 1.0)
        returns[t] = R
    
    return returns


# -------------------------
# Instantiate model & optimizers
# -------------------------
model = SharedActorCritic(state_dim, action_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_ACTOR)  # we use single optimizer for both heads
# optionally separate optimizers: actor_params, critic_params


# -------------------------
# Main A2C loop
# -------------------------
start_time = time.time()
episode_rewards = []
global_step = 0

# We'll run until NUM_UPDATES updates; each update consumes ~N_STEPS env steps (or fewer near episode endings)
for update in range(1, NUM_UPDATES + 1):
    # storage for batch
    states = []
    actions = []
    log_probs = []
    entropies = []
    rewards = []
    dones = []
    values = []
    
    # Collect up to N_STEPS interactions (may break earlier if episode ends)
    state, _ = env.reset()
    state = preprocess_state(state)
    
    # If continuing an episode across updates, you'd want to keep state; here we reset each update for simplicity
    for step in range(N_STEPS):
        s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)  # shape (1, state_dim)
        mean, std, value = model.forward(s_tensor)
        # sample an action and get corrected log prob and entropy
        action_env, logp, entropy, tanh_act, pre_tanh = model.sample_action(s_tensor)
        # env step expects shape (action_dim,) numpy float array
        next_state, reward, terminated, truncated, _ = env.step(action_env.flatten())
        done = bool(terminated or truncated)
        reward = float(reward)  # ensure scalar
        
        # store components
        states.append(state)
        actions.append(action_env.flatten())
        log_probs.append(logp.squeeze(0))      # shape scalar tensor
        entropies.append(entropy.squeeze(0))
        rewards.append(reward)
        dones.append(done)
        values.append(value.squeeze(0))        # V(s_t) tensor
        
        global_step += 1
        state = preprocess_state(next_state)
        
        if done:
            # If episode ended prematurely, we break and bootstrap last_value as 0
            break
    
    # bootstrap value for final state (s_T). If last step was terminal, V(s_T)=0
    if done:
        last_value = 0.0
    else:
        s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, _, last_v = model.forward(s_tensor)
            last_value = float(last_v.squeeze(0).cpu().numpy())
    
    # Compute n-step returns (bootstrapped)
    returns_np = compute_nstep_returns(rewards, last_value, GAMMA, dones)
    
    # Convert lists to tensors for batch update
    returns = torch.tensor(returns_np, dtype=torch.float32, device=DEVICE)         # shape (T,)
    log_probs_tensor = torch.stack(log_probs).to(DEVICE)                          # shape (T,)
    entropies_tensor = torch.stack(entropies).to(DEVICE)                          # shape (T,)
    values_tensor = torch.stack(values).to(DEVICE)                                # shape (T,)
    
    # Advantage: A = R^{(n)} - V(s)
    advantages = returns - values_tensor.detach()
    
    # Optionally normalize advantages (recommended)
    if NORMALIZE_ADV:
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
    
    # Actor loss: negative expected advantage-weighted log-prob + entropy bonus
    actor_loss = - (log_probs_tensor * advantages).mean() - ENTROPY_COEF * entropies_tensor.mean()
    
    # Critic loss: MSE between returns (bootstrapped) and V(s)
    critic_loss = nn.functional.mse_loss(values_tensor, returns)
    
    # Total loss: weighted sum (value coef scales critic loss)
    total_loss = actor_loss + VALUE_COEF * critic_loss
    
    # Backpropagate and update
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    
    # Logging / bookkeeping: compute episodic returns if episode ended in this batch
    # We'll append total reward of last episode if done; else accumulate (approx)
    if done:
        episode_rewards.append(sum(rewards))
    else:
        # if episode not finished, append partial return as proxy
        episode_rewards.append(sum(rewards) + (GAMMA ** len(rewards)) * last_value)
    
    # trim history
    if len(episode_rewards) > 100:
        episode_rewards.pop(0)
    
    if update % 10 == 0 or update == 1:
        avg100 = float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else 0.0
        elapsed = time.time() - start_time
        print(f"Update {update:4d} | Steps {global_step:6d} | AvgReturn(100): {avg100:8.3f} | "
              f"ActorLoss: {actor_loss.item():.4f} | CriticLoss: {critic_loss.item():.4f} | Time: {elapsed:.1f}s")


# Clean up
env.close()
print("A2C training complete ✅")


# -------------------------
# Deterministic evaluation: use policy mean (no sampling)
# -------------------------
def evaluate(model, env_name=ENV_NAME, n_episodes=5, render=False):
    eval_env = gym.make(env_name, render_mode="human" if render else None)
    returns = []
    for _ in range(n_episodes):
        s, _ = eval_env.reset()
        s = preprocess_state(s)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < MAX_EPISODE_LEN:
            s_tensor = torch.from_numpy(s).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mean, std, _ = model.forward(s_tensor)
                # deterministic action = tanh(mean)
                action = torch.tanh(mean)
                action_env = (action * action_scale + action_bias).cpu().numpy().flatten()
            s, r, terminated, truncated, _ = eval_env.step(action_env)
            r = float(r)
            total += r
            s = preprocess_state(s)
            done = bool(terminated or truncated)
            steps += 1
            if render:
                eval_env.render()
        returns.append(total)
    eval_env.close()
    return np.mean(returns), np.std(returns)


mean_r, std_r = evaluate(model, n_episodes=5, render=False)
print(f"Evaluation (deterministic mean) over 5 eps: {mean_r:.2f} ± {std_r:.2f}")

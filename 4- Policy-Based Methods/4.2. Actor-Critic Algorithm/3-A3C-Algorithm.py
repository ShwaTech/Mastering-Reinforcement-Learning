"""
A3C (Asynchronous Advantage Actor-Critic) for continuous actions (Pendulum-v1)

- Shared global model (actor+critic). Each worker keeps a local copy,
collects up to t_max steps, computes gradients and applies them to the shared model.
- Actor: Gaussian policy (mean, learnable log_std per action dim).
- Actions are tanh-squashed; log-prob corrected by Jacobian of tanh.
- Critic: state-value V(s). Critic trained with n-step bootstrapped returns.
- This is a pedagogical, careful implementation (shape checks, numeric stability).
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Normal


# ----------------------------
# Configuration & hyperparams
# ----------------------------
ENV_NAME = "Pendulum-v1"
NUM_WORKERS = max(1, mp.cpu_count() - 1)   # number of worker processes
T_MAX = 5               # n-step length for each worker rollout (typical A3C uses 5)
GAMMA = 0.99
LR = 7e-4               # learning rate (original A3C used RMSProp; Adam works too)
ENTROPY_COEF = 1e-3     # entropy bonus weight
VALUE_COEF = 0.5        # critic loss weight in total loss
MAX_UPDATES = 2000      # number of global updates (approx)
GRAD_CLIP = 0.5
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
SEED = 1


# reproducible seeds
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------
# Utility functions
# ----------------------------
def preprocess_state(state, state_dim):
    """Return flattened float32 1-D numpy array of length state_dim."""
    arr = np.asarray(state, dtype=np.float32).flatten()
    assert arr.shape[0] == state_dim, f"State shape mismatch {arr.shape}"
    return arr

def compute_nstep_bootstrap_returns(rewards, last_value, gamma, dones):
    """
    Compute bootstrapped returns R_t^{(n)} for a rollout of length T:
      R_t = r_t + γ*r_{t+1} + ... + γ^{T-1-t}*r_{T-1} + γ^{T-t} V(s_T) * (1 - done_T)
    `dones` is list of booleans whether step t ended episode.
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    R = float(last_value)
    for t in reversed(range(T)):
        R = rewards[t] + gamma * R * (0.0 if dones[t] else 1.0)
        returns[t] = R
    return returns


# ----------------------------
# Shared model: Shared Actor-Critic
# ----------------------------
class SharedActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        # shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # actor mean head
        self.mean = nn.Linear(hidden, action_dim)
        # global learnable log_std parameter per action dim (stable)
        self.log_std_param = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        # critic head
        self.value = nn.Linear(hidden, 1)
    
    
    def forward(self, x):
        """
        x: (batch, state_dim)
        returns:
            mean: (batch, action_dim)
            std: (action_dim,)  (expanded per-batch outside)
            value: (batch,)
        """
        h = self.backbone(x)
        mean = self.mean(h)                # (batch, action_dim)
        log_std = torch.clamp(self.log_std_param, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)           # (action_dim,)
        value = self.value(h).squeeze(-1)  # (batch,)
        return mean, std, value
    
    
    def act_and_eval(self, state_tensor, action_scale, action_bias):
        """
        Sample action using reparameterization, apply tanh squash and scale to env range.
        Returns:
            action_env: numpy array (action_dim,)
            log_prob: torch tensor scalar (batch,)
            entropy: torch tensor scalar (batch,)
            value: torch tensor (batch,)
        """
        mean, std, value = self.forward(state_tensor)  # mean (1,A), std (A,), value (1,)
        # reparameterize
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std.unsqueeze(0)  # (1,A)
        tanh_action = torch.tanh(pre_tanh)        # (-1,1)
        action_env = (tanh_action * action_scale + action_bias).detach().cpu().numpy().flatten()
        
        # compute corrected log_prob: Normal(pre_tanh; mean, std) - sum log(1 - tanh^2 + eps)
        normal = Normal(mean, std.unsqueeze(0))
        logp_pre = normal.log_prob(pre_tanh).sum(dim=-1)      # (1,)
        log_det = torch.log(1 - tanh_action.pow(2) + 1e-6).sum(dim=-1)  # (1,)
        log_prob = logp_pre - log_det                         # (1,)
        entropy = normal.entropy().sum(dim=-1)                # (1,)
        return action_env, log_prob, entropy, value, pre_tanh


# ----------------------------
# Worker process
# ----------------------------
def worker_process(global_model, optimizer, counter, lock, worker_id, config):
    """
    Each worker:
        - Creates its own env
        - Maintains a local copy of the model parameters (by syncing from global model)
        - Collects up to T_MAX steps, computes n-step returns, computes gradients locally,
        and applies gradients to the global model by copying local grads into global and optimizer.step().
    Notes:
        - optimizer is the shared optimizer passed from the main process (state in shared mem).
        - counter is a mp.Value counting global updates/episodes.
    """
    # Unpack config (to avoid closure capturing many globals)
    T_max = config["T_MAX"]
    gamma = config["GAMMA"]
    env_name = config["ENV_NAME"]
    seed = config["SEED"] + worker_id   # different seed per worker
    entropy_coef = config["ENTROPY_COEF"]
    value_coef = config["VALUE_COEF"]
    grad_clip = config["GRAD_CLIP"]
    max_updates = config["MAX_UPDATES"]
    
    # Local environment per worker (no rendering)
    env = gym.make(env_name, render_mode="human")
    env.reset(seed=seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0
    
    # local model copy (same architecture)
    local_model = SharedActorCritic(state_dim, action_dim)
    local_model.load_state_dict(global_model.state_dict())  # initial sync
    
    # Put local model in training mode
    local_model.train()
    
    # training loop: continue until global counter reaches max_updates
    while True:
        with lock:
            # read global counter atomically
            if counter.value >= max_updates:
                break
        
        # sync local model with global model before rollout
        local_model.load_state_dict(global_model.state_dict())
        
        # storage for rollout
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        entropies = []
        dones = []
        
        # reset env if starting fresh
        state, _ = env.reset()
        state = preprocess_state(state, state_dim)
        
        # Collect up to T_max steps or until done
        for t in range(T_max):
            s_tensor = torch.from_numpy(state).float().unsqueeze(0)  # shape (1, state_dim)
            # sample action from local_model (to avoid globally interfering sampling)
            action_env, log_prob, entropy, value, pre_tanh = local_model.act_and_eval(s_tensor, action_scale, action_bias)
            next_state, reward, terminated, truncated, _ = env.step(action_env)
            done = bool(terminated or truncated)
            reward = float(reward)
            
            # store trajectory elements
            states.append(state)
            actions.append(action_env)
            log_probs.append(log_prob.squeeze(0))
            rewards.append(reward)
            values.append(value.squeeze(0))
            entropies.append(entropy.squeeze(0))
            dones.append(done)
            
            state = preprocess_state(next_state, state_dim)
            
            if done:
                # episode ended; break to bootstrap with last_value=0.0
                break
        
        # bootstrap value for last state
        if dones[-1]:
            last_value = 0.0
        else:
            s_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                _, _, last_value_t = local_model.forward(s_tensor)
                last_value = float(last_value_t.squeeze(0).item())
        
        # compute n-step bootstrapped returns
        returns_np = compute_nstep_bootstrap_returns(rewards, last_value, gamma, dones)
        returns = torch.tensor(returns_np, dtype=torch.float32)   # shape (T,)
        
        # convert lists to tensors
        log_probs_tensor = torch.stack(log_probs)   # shape (T,)
        values_tensor = torch.stack(values)         # shape (T,)
        entropies_tensor = torch.stack(entropies)   # shape (T,)
        
        # advantages
        advantages = returns - values_tensor.detach()
        
        # losses (mean over T)
        actor_loss = - (log_probs_tensor * advantages).mean() - entropy_coef * entropies_tensor.mean()
        critic_loss = (returns - values_tensor).pow(2).mean()
        total_loss = actor_loss + value_coef * critic_loss
        
        # zero local grads
        local_model.zero_grad()
        # backprop on local model
        total_loss.backward()
        
        # copy local gradients to global model (manual)
        # ensure global params require grad, then assign grads
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            if local_param.grad is None:
                global_param.grad = None
            else:
                # copy gradient data to global param
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad.copy_(local_param.grad)
        
        # clip grads on global model
        torch.nn.utils.clip_grad_norm_(global_model.parameters(), grad_clip)
        
        # step the shared optimizer (applies asynchronously from multiple workers)
        optimizer.step()
        optimizer.zero_grad()
        
        # increment global counter (safely)
        with lock:
            counter.value += 1
            # optional: save checkpoint occasionally
            if counter.value % 200 == 0:
                print(f"[Worker {worker_id}] Completed checkpoint at step {counter.value}")
    
    env.close()
    print(f"[Worker {worker_id}] Finished. Exiting.")


# ----------------------------
# Main function to start A3C
# ----------------------------
def main():
    mp.set_start_method("spawn", force=True)  # spawn is safe cross-platform
    env = gym.make(ENV_NAME, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()
    
    # create shared global model
    global_model = SharedActorCritic(state_dim, action_dim)
    global_model.share_memory()   # put parameters in shared memory
    
    # shared optimizer (we will share optimizer state by creating it in main and passing to workers)
    optimizer = optim.Adam(global_model.parameters(), lr=LR)
    # NB: optimizer state is not automatically in shared memory; for production you may need a
    # shared optimizer implementation (e.g., SharedAdam). Here we pass the optimizer object
    # to child processes — this works for many simple use-cases, but may be non-ideal on some platforms.
    # Alternative: implement shared RMSprop or SharedAdam (longer).
    
    # shared counter and lock for synchronization
    counter = mp.Value('i', 0)    # integer counter of updates
    lock = mp.Lock()
    
    # config dict to pass to workers
    config = {
        "T_MAX": T_MAX,
        "GAMMA": GAMMA,
        "ENV_NAME": ENV_NAME,
        "SEED": SEED,
        "ENTROPY_COEF": ENTROPY_COEF,
        "VALUE_COEF": VALUE_COEF,
        "GRAD_CLIP": GRAD_CLIP,
        "MAX_UPDATES": MAX_UPDATES
    }
    
    # spawn worker processes
    processes = []
    for worker_id in range(NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(global_model, optimizer, counter, lock, worker_id, config))
        p.start()
        processes.append(p)
        time.sleep(0.1)  # stagger startups slightly
    
    # wait for processes to finish
    for p in processes:
        p.join()
    
    # training finished.
    print(f"A3C training finished. ✅ Total updates: {counter.value}")
    
    # Optional: quick deterministic evaluation using mean action
    evaluate_policy(global_model, n_eps=5, render=False)


# ----------------------------
# Evaluation helper
# ----------------------------
def evaluate_policy(model, n_eps=5, render=True):
    eval_env = gym.make(ENV_NAME, render_mode="human" if render else None)
    state_dim = eval_env.observation_space.shape[0]
    action_low = float(eval_env.action_space.low[0])
    action_high = float(eval_env.action_space.high[0])
    scale = (action_high - action_low) / 2.0
    bias = (action_high + action_low) / 2.0
    
    model.eval()
    returns = []
    for ep in range(n_eps):
        s, _ = eval_env.reset()
        s = preprocess_state(s, state_dim)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < 1000:
            s_tensor = torch.from_numpy(s).float().unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model.forward(s_tensor)
                # deterministic mean; apply tanh and scale
                action = torch.tanh(mean)
                action_env = (action * scale + bias).cpu().numpy().flatten()
            s, r, terminated, truncated, _ = eval_env.step(action_env)
            r = float(r)
            total += r
            s = preprocess_state(s, state_dim)
            done = bool(terminated or truncated)
            steps += 1
            if render:
                eval_env.render()
        returns.append(total)
        print(f"[Eval] Episode {ep+1} return: {total:.2f}")
    
    eval_env.close()
    print(f"[Eval] Mean return over {n_eps} episodes: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    return np.mean(returns), np.std(returns)


# ----------------------------
# Run as script
# ----------------------------
if __name__ == "__main__":
    main()

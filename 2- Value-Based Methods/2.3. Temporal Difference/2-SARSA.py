import gymnasium as gym
import numpy as np


# Create FrozenLake environment (deterministic because is_slippery=False)
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)


class SARSA():
    """
    SARSA (State–Action–Reward–State–Action) Algorithm
    On-policy Temporal-Difference (TD) control algorithm.
    Learns the Q-function by updating using the *actual action taken*.
    """

    def __init__(self, num_iterations, epsilon, discount, alpha):
        # Total number of training episodes
        self.num_iterations = num_iterations
        
        # Number of possible states in the environment
        self.num_states = env.observation_space.n
        
        # Number of possible actions in the environment
        self.num_actions = env.action_space.n
        
        # Exploration rate (epsilon-greedy policy)
        self.epsilon = epsilon
        
        # Discount factor (γ) for future rewards
        self.discount = discount
        
        # Learning rate (α)
        self.alpha = alpha
    
    def MainAlgorithm(self):
        # Initialize Q-table with zeros
        # Q[state, action] → expected return (value) for taking action in state
        Q = np.zeros((self.num_states, self.num_actions))
        
        # Run for the specified number of training episodes
        for i in range(self.num_iterations):
            # Reset environment at start of episode
            curr_state = env.reset()[0]

            # Choose initial action using epsilon-greedy strategy
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0, self.num_actions)  # Explore
            else:
                action = np.argmax(Q[curr_state])                # Exploit (greedy)
            
            done = False

            # Play one full episode
            while not done:
                # Take the action in the environment
                next_state, R, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Choose next action using epsilon-greedy (on-policy)
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(0, self.num_actions)  # Explore
                else:
                    next_action = np.argmax(Q[next_state])                # Exploit
                
                # SARSA Update Rule:
                # Q(s,a) ← Q(s,a) + α * [R + γ * Q(s',a') - Q(s,a)]
                Q[curr_state, action] += self.alpha * (
                    R + self.discount * Q[next_state, next_action] - Q[curr_state, action]
                )

                # Move to next state and action
                curr_state = next_state
                action = next_action

        # After training, derive the greedy policy from Q
        policy = np.argmax(Q, axis=1)
        return policy


# --------------------- Training ---------------------

# Hyperparameters
n_iterations = 100_000   # Number of episodes
epsilon = 0.8            # Exploration rate
discount = 0.9           # Discount factor (γ)
alpha = 0.85             # Learning rate (α)

# For reproducibility
np.random.seed(1)

# Create SARSA agent and train it
sarsa = SARSA(n_iterations, epsilon, discount, alpha)
policy = sarsa.MainAlgorithm()

# Close training environment
env.close()

# --------------------- Testing ---------------------
print("------------- Testing -------------")
print(policy)

# Recreate environment for visualization
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)

# Reset to start state
curr_state = env.reset()[0]

done = False
reward = 0

while not done:
    # Select action based on the learned greedy policy
    action = policy[curr_state]

    # Take action in environment
    next_state, R, terminated, truncated, _ = env.step(action=action)

    env.render()  # Render environment to visualize actions

    # Episode ends if terminated (goal/hole) or truncated (time limit)
    done = terminated or truncated

    reward = R  # Store reward (1 if goal reached, else 0)

    curr_state = next_state  # Move to next state

env.close()

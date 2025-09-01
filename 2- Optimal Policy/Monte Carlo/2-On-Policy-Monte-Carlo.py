import gymnasium as gym
import numpy as np

## Deterministic Environment and Only For Training (Runs in Background)
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)

class OnPolicyMonteCarlo():
    def __init__(self, num_iterations, epsilon, e_decay, min_e):
        self.num_iterations = num_iterations
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.epsilon = epsilon
        self.e_decay = e_decay
        self.min_e = min_e
    
    def MonteCarloAlgorithm(self):
        ## Initialized With Zero and Random Policy
        total_return_s_a = np.zeros((self.num_states, self.num_actions))
        N_s_a = np.zeros((self.num_states, self.num_actions))
        Q = np.zeros((self.num_states, self.num_actions))
        policy = np.random.randint(0, self.num_actions, self.num_states)

        for i in range(self.num_iterations):
            episode = self.GenerateEpisode(policy)
            rewards = [ x[2] for x in episode ]
            visited_states_in_episode = []  # For First Visit Algorithm

            index = 0
            for step in episode:  # [(state, action, reward), (state, action, reward), ...]
                state = step[0]
                if not state in visited_states_in_episode:
                    visited_states_in_episode.append(state)
                    R = sum(rewards[index:])

                    action = step[1]
                    total_return_s_a[state, action] += R
                    N_s_a[state, action] += 1

                    Q[state, action] = total_return_s_a[state, action] / N_s_a[state, action]
                
                index += 1
            
            policy = np.argmax(Q, axis=1)

            if self.epsilon > self.min_e:
                self.epsilon *= self.e_decay
        
        return policy
    
    def GenerateEpisode(self, policy):
        # state = [(18, 7, False), {}]
        # x = state[0][0]
        # y = state[0][1]
        # z = int(state[0][2])
        # return policy[x][y][z]

        episode = []    ## [(current_state, action, reward)]
        cur_state = env.reset()[0]

        done = False

        while not done:
            action = policy[cur_state]

            ## Epsilon Greedy
            if np.random.rand() < self.epsilon:
                ## Exploration (Take a Random Action)
                action = np.random.randint(0, self.num_actions)

            next_state, r, terminated, trunkated, _ = env.step(action)

            transition = (cur_state, action, r)
            episode.append(transition)

            done = terminated or trunkated

            cur_state = next_state
        
        return episode


n_iterations = 10_000
epsilon = 1
epsilon_decay = 0.999975
min_epsilon = 0.001

np.random.seed(1)

OPMC = OnPolicyMonteCarlo(n_iterations, epsilon, epsilon_decay, min_epsilon)

policy = OPMC.MonteCarloAlgorithm()

env.close()

print("Testing...")
print(policy)

## Deterministic Environment For Testing (Runs in the Foreground and Human See it)
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)

cur_state = env.reset()[0]

done = False
while not done:
    action = policy[cur_state]

    next_state, r, terminated, trunkated, _ = env.step(action)

    env.render()

    done = terminated or trunkated

    cur_state = next_state


env.close()


import gymnasium as gym
import numpy as np

# ## deterministic Environment
# env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)

## Stochastic Environment 
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)


class PolicyIteration():
    def __init__(self, n_iterations, discount_factor):
        self.n_iterations = n_iterations
        self.discount_factor = discount_factor
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        ## Initialize The Value of All States With Zero
        self.values_table = [0] * self.num_states
        self.optimal_policy = list(np.random.randint(0, self.num_actions, self.num_states))
    
    def get_optimal_policy(self):
        for i in range(self.n_iterations):
            for state_i in range(self.num_states):

                ## Get Action By Action From Our Randomized self.optimal_policy
                action = self.optimal_policy[state_i]

                ## Transition Probability For action in state_i
                ## [ (prob, state, reward), (prob, state, reward), ... ]
                trans_prob = env.unwrapped.P[state_i][action]

                ## Extract The Prob. From trans_prob
                prob = np.array([ x[0] for x in trans_prob ])

                ## Extract The Reward From trans_prob
                reward = np.array([ x[2] for x in trans_prob ])

                ## Calculate The Second Part Of the Equation
                R = reward + self.discount_factor * np.array([ self.values_table[x[1]] for x in trans_prob ])

                ## Final Result is The V -> Value Function
                V = sum(prob * R)

                ## Assign New Values V To Our Zero-Initiated values_table
                self.values_table[state_i] = V


                #### Now Getting Optimal Policy
                actions_Q = [0] * self.num_actions

                for action in range(self.num_actions):
                    ## Transition Probability For Every New action in state_i
                    ## [ (prob, state, reward), (prob, state, reward), ... ]
                    trans_prob = env.unwrapped.P[state_i][action]

                    ## Extract The Prob. From trans_prob
                    prob = np.array([ x[0] for x in trans_prob ])

                    ## Extract The Reward From trans_prob
                    reward = np.array([ x[2] for x in trans_prob ])

                    ## Calculate The Second Part Of the Equation
                    R = reward + self.discount_factor * np.array([ self.values_table[x[1]] for x in trans_prob ])

                    ## Final Result is The Q -> Q-Function
                    Q = sum(prob * R)

                    actions_Q[action] = Q
                
                self.optimal_policy[state_i] = np.argmax(actions_Q)
        
        return self.optimal_policy


cur_state = env.reset()

n_iterations = 1000
discount_factor = 0.9
np.random.seed(1)

app = PolicyIteration(n_iterations, discount_factor)

optimal_policy = app.get_optimal_policy()

print(f"Optimal Policy -->\n{optimal_policy}")

s = 0
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(int(optimal_policy[s]))
    s = int(obs)
    done = terminated or truncated
    env.render()

env.close()


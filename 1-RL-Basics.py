import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="human")

cur_state = env.reset()

print(f"Current State => {cur_state}")

## Number of States
print("Number_of_States : ", env.observation_space)

## Number of Actions
print("Number_of_States : ", env.action_space)

## Return the prob. of Going to each state (Stochastic Environment)
# States -> Grid Numbers Starts From 0 To 15 ??
# 0 --> Left
# 1 --> Down
# 2 --> Right
# 3 --> Up
print("ST. Transition Probability (State 0, Action 1) =>\n", env.unwrapped.P[0][1])

## Cahnge Our Environment To Be Deterministic 
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
print("Det. Transition Probability (State 0, Action 1) =>\n", env.unwrapped.P[0][1])

## Another Example ???
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)
print("ST. Transition Probability (State 6, Action 2) =>\n", env.unwrapped.P[6][2])

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
print("Det. Transition Probability (State 6, Action 2) =>\n", env.unwrapped.P[6][2])

## Rendering Or Updating The Environment Needs it To Be Reseted First ??
env.reset()
env.render()

## Returns (next_state, reward, terminated, truncated, trans_prob to new_state from prev_state)
# Done = truncated or terminated
state = env.step(1)
env.render()
print(f"Updated State : {state}")


## Sample Random Action
random_action = env.action_space.sample()  # -> int
print(f"Random Action : {random_action}")


## Generate a Random Episode 
state = env.reset()
print("Time Step 0 : ")
env.render()

num_timesteps = 20
for t in range(num_timesteps):
    random_action = env.action_space.sample()
    new_state, reward, terminated, truncated, info = env.step(random_action)
    print(f"Time Step : {t+1}")
    
    env.render()
    
    done = truncated or terminated
    if done:
        break

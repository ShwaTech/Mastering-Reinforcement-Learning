import random
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam
import gymnasium as gym
import ale_py

# Register Atari environments (provided by the ALE library)
gym.register_envs(ale_py)

# Initialize the Ms. Pac-Man Atari environment
env = gym.make("ALE/MsPacman-v5")

# Get initial environment state to extract info
state, _ = env.reset()

# Resize target frame dimensions and set channel depth to 1 (grayscale)
state_size = (88, 80, 1)

# Number of available actions (output dimension of the network)
action_size = env.action_space.n

# Compute the mean color value for the background color to remove it later
color = np.array([210, 164, 74]).mean()


# ===========================
#  Preprocessing Function
# ===========================
def preprocess_state(state):
    """Convert raw RGB Atari frame into a compact grayscale format suitable for CNN input."""
    image = state[1:176:2, ::2]              # Crop the game area and downsample every 2 pixels
    image = image.mean(axis=2)               # Convert to grayscale by averaging RGB channels
    image[image == color] = 0                # Remove the background color (make it black)
    image = (image - 128.0) / 128.0 - 1.0    # Normalize pixel values roughly to range [-2, 0]
    image = np.expand_dims(image.reshape(state_size), axis=0)  # Add batch dimension (1, 88, 80, 1)
    return image


# ===========================
#  DQN Class (Deep Q-Network)
# ===========================
class DQN:
    def __init__(self, state_size, action_size):
        # Dimensions of input (preprocessed frame)
        self.state_size = state_size

        # Number of actions (outputs)
        self.action_size = action_size

        # Experience replay memory to store (state, action, reward, next_state, done)
        self.replay_buffer = deque(maxlen=5000)

        # Discount factor for future rewards (γ)
        self.gamma = 0.9

        # Exploration rate for epsilon-greedy strategy
        self.epsilon = 0.8

        # Frequency of updating target network
        self.update_rate = 1000

        # Build the main (training) and target (stabilization) neural networks
        self.main_network = self.build_network()
        self.target_network = self.build_network()

        # Initialize both networks with identical weights
        self.target_network.set_weights(self.main_network.get_weights())

    def reset(self):
        """Reload a saved model from disk and synchronize target network."""
        self.main_network = load_model("Atari_3.keras")
        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())

    def build_network(self):
        """Construct a CNN-based Q-network that maps frames → Q-values for each action."""
        model = Sequential([
            # First convolutional layer: detects edges and shapes
            Conv2D(32, (8, 8), strides=4, padding='same', activation='relu', input_shape=self.state_size),
            # Second convolutional layer: detects more complex patterns
            Conv2D(64, (4, 4), strides=2, padding='same', activation='relu'),
            # Third convolutional layer: deeper feature extraction
            Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
            # Flatten the 3D feature maps to 1D for fully connected layers
            Flatten(),
            # Dense hidden layer: combines learned features into Q-value estimates
            Dense(512, activation='relu'),
            # Output layer: one neuron per possible action (Q-values)
            Dense(self.action_size, activation='linear')
        ])
        # Compile network with MSE loss and Adam optimizer (standard for DQN)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.00025))
        return model

    def store_transition(self, state, action, reward, next_state, done):
        """Store a single experience tuple in the replay buffer for later sampling."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state):
        """Select an action using epsilon-greedy strategy for exploration/exploitation balance."""
        # With probability epsilon → choose random action (exploration)
        if random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        # Otherwise choose action with the highest predicted Q-value (exploitation)
        Q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(Q_values[0])

    def train(self, batch_size):
        """Train the main network using random samples from replay memory."""
        # Randomly sample a minibatch of experiences
        minibatch = random.sample(self.replay_buffer, batch_size)

        # For each experience tuple, update the Q-value estimate
        for state, action, reward, next_state, done in minibatch:
            # Compute the target Q-value (based on target network prediction)
            target_Q = reward
            if not done:
                # If episode not finished, add discounted max next-state Q-value
                target_Q += self.gamma * np.amax(self.target_network.predict(next_state, verbose=0))

            # Get the current Q-values from the main network
            Q_values = self.main_network.predict(state, verbose=0)
            # Update the chosen action's Q-value towards the target
            Q_values[0][action] = target_Q

            # Train (fit) the network for one gradient update step
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)

    def update_target_network(self):
        """Synchronize target network weights with main network (stabilizes training)."""
        self.target_network.set_weights(self.main_network.get_weights())

    def save_network(self, i):
        """Save the trained main network to disk for future use."""
        self.main_network.save(f"Atari_{i}.keras")

    def play_game(self):
        """Load a trained model and play one full episode in the environment."""
        model = load_model("Atari_6.keras")
        done = False

        state, _ = env.reset()
        env.render()

        # Loop until game over
        while not done:
            # Always pick the action with the highest Q-value (greedy play)
            action = np.argmax(model.predict(preprocess_state(state), verbose=0)[0])
            # Perform the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Render the environment for visualization
            env.render()
            # Move to next state
            state = next_state

        env.close()


# ===========================
#  Main Training Loop
# ===========================
if __name__ == "__main__":
    # Number of full episodes to train
    num_episodes = 12

    # Maximum number of steps per episode
    num_timesteps = 20000

    # Number of samples per training batch
    batch_size = 8

    # Initialize Deep Q-Network agent
    dqn = DQN(state_size, action_size)

    # Optional: Uncomment to watch pretrained model play
    # dqn.play_game()

    time_step = 0  # Counts total environment steps

    # Iterate over episodes
    for i in range(num_episodes):
        total_reward = 0
        done = False

        # Reset environment and preprocess initial frame
        state, _ = env.reset()
        state = preprocess_state(state)

        # Iterate over steps in the episode
        for t in range(num_timesteps):
            print(f"Episode {i} | Step {t}/{num_timesteps}")

            # Increment global time step counter
            time_step += 1

            # Update target network periodically to stabilize learning
            if time_step % dqn.update_rate == 0:
                dqn.update_target_network()

            # Select action using epsilon-greedy strategy
            action = dqn.epsilon_greedy(state)

            # Execute action and observe outcome
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Preprocess next frame for neural network input
            next_state = preprocess_state(next_state)

            # Store experience in replay buffer
            dqn.store_transition(state, action, reward, next_state, done)

            # Move agent to next state
            state = next_state
            total_reward += reward

            # If episode ends (agent dies or level finishes)
            if done:
                print(f"✅ Episode {i} finished with Return: {total_reward}")
                break

            # Train DQN once enough experiences are collected
            if len(dqn.replay_buffer) > batch_size:
                dqn.train(batch_size)

        # Save model weights every episode for progress tracking
        dqn.save_network(i + 4)

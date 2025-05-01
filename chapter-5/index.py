import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="human")
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1 # Learning rate
gamma = 0.99 # Discount factor
epsilon = 0.1 # Exploration rate

# Training loop
for episode in range(1000):
    state = env.reset()[0] # Get the initial state
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample() # Explore
        else:
            action = np.argmax(q_table[state])# Exploit
        # Unpack all five values from step()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated) # Update done

        # Update Q-table
        q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
        state = next_state

# Test the trained agent
num_test_episodes = 10
for episode in range(num_test_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += reward
        state = next_state
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()

import numpy as np

class PPO:
    def __init__(self, action_size):
        self.weights = np.random.rand(action_size) * 0.01  # Random initial weights
    
    def select_action(self, state):
        action_probs = np.exp(self.weights) / np.sum(np.exp(self.weights))  # Softmax for action probabilities
        return np.random.choice(len(action_probs), p=action_probs)  # Sample action

    def update(self, states, actions, rewards, epsilon=0.2):
        old_probs = np.exp(self.weights[actions]) / np.sum(np.exp(self.weights))  # Old probs
        advantages = rewards - np.mean(rewards)
        ratio = np.exp(self.weights[actions] - old_probs)
        clipped_advantage = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
        loss = -np.mean(np.minimum(ratio * advantages, clipped_advantage))
        grad = np.dot(np.array(states).T, (rewards - old_probs) * advantages)
        self.weights += 0.01 * grad  # Update weights

# Simple Environment for personalized offers (3 actions: 0 -> No discount, 1 -> 5% discount, 2 -> 10% discount)
ecommerce_data = [0, 1, 0, 1, 0]  # 0 = not purchased, 1 = purchased
ppo = PPO(action_size=3)

# Training loop
for _ in range(5):
    states = np.random.random(5)  # Random states (for simplicity)
    actions = [ppo.select_action(state) for state in states]
    rewards = [10 if action == 2 else -5 for action in actions]  # Rewards based on action
    ppo.update(states, actions, rewards)

    # Display decision-making output:
    for i, action in enumerate(actions):
        if action == 0:
            print(f"Offer: No discount, Reason: User didn't purchase or already offered.")
        elif action == 1:
            print(f"Offer: 5% discount, Reason: User might be on the edge of purchasing.")
        elif action == 2:
            print(f"Offer: 10% discount, Reason: Strong chance of user purchasing with a bigger discount.")

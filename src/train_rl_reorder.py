import numpy as np
from rl_inventory_env import InventoryEnv

# ======================
# ENVIRONMENT
# ======================
env = InventoryEnv()

MAX_INV = 200
N_ACTIONS = env.action_space.n

# Q-table: inventory x demand x action
q_table = np.zeros((MAX_INV + 1, MAX_INV + 1, N_ACTIONS))

# ======================
# HYPERPARAMETERS
# ======================
alpha = 0.1        # learning rate
gamma = 0.95       # discount factor
epsilon = 1.0      # exploration
epsilon_decay = 0.995
epsilon_min = 0.05

episodes = 500

# ======================
# TRAINING LOOP
# ======================
for ep in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # CURRENT STATE (CLIPPED)
        inv = int(np.clip(state[0], 0, MAX_INV))
        dem = int(np.clip(state[1], 0, MAX_INV))

        # ACTION (ε-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[inv, dem])

        # TAKE ACTION
        next_state, reward, done, _, _ = env.step(action)

        # NEXT STATE (CLIPPED)
        next_inv = int(np.clip(next_state[0], 0, MAX_INV))
        next_dem = int(np.clip(next_state[1], 0, MAX_INV))

        # Q-LEARNING UPDATE
        best_next_q = np.max(q_table[next_inv, next_dem])
        q_table[inv, dem, action] += alpha * (
            reward + gamma * best_next_q - q_table[inv, dem, action]
        )

        state = next_state
        total_reward += reward

    # DECAY EPSILON
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % 50 == 0:
        print(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

print("✅ RL Training Completed")

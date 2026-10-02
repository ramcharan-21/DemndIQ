import numpy as np
import torch
from rl_inventory_env import InventoryEnv
from dqn_agent import DQNAgent

env = InventoryEnv()

state_dim = 4
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

episodes = 200
target_update_freq = 10
MODEL_DIR = "models"

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    if ep % target_update_freq == 0:
        agent.update_target()

    if ep % 10 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
    if (ep + 1) % 50 == 0:
        torch.save(agent.model.state_dict(), f"{MODEL_DIR}/dqn_model.pt")
        print(f"✅ Checkpoint saved at episode {ep+1}")

print("✅ DQN Training Completed")

import torch

MODEL_DIR = "models"
torch.save(agent.model.state_dict(), f"{MODEL_DIR}/dqn_model.pt")
print("✅ Final DQN model saved")


import numpy as np
import torch
from rl_inventory_env import InventoryEnv
from dqn_agent import DQNAgent
from pathlib import Path

def evaluate():
    print("Starting RL Policy Evaluation...")
    
    # Setup
    env = InventoryEnv()
    state_dim = 4
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    # Load Model
    model_path = Path("models/dqn_model.pt")
    if model_path.exists():
        print(f"Loading model from {model_path}")
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval() # Set to eval mode
    else:
        print(f"Warning: {model_path} not found. Running with random weights.")

    # Evaluation Config
    num_episodes = 2
    total_rewards = []
    avg_inventories = []
    reorder_counts = [] # Count of steps where action > 0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        ep_inventory = []
        ep_reorders = 0
        
        while not done:
            # Greedy action
             # Custom Greedy Act (bypass epsilon)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_vals = agent.model(state_tensor)
                action = torch.argmax(q_vals).item()
            
            # Step
            next_state, reward, done, _, _ = env.step(action)
            
            # Metrics
            ep_reward += reward
            ep_inventory.append(state[0]) # state[0] is inventory
            if action > 0:
                ep_reorders += 1
                
            state = next_state
            
        total_rewards.append(ep_reward)
        avg_inventories.append(np.mean(ep_inventory))
        reorder_counts.append(ep_reorders)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes}: Reward={ep_reward:.1f}, AvgInv={np.mean(ep_inventory):.1f}")

    # Summary Stats
    mean_reward = np.mean(total_rewards)
    mean_inv = np.mean(avg_inventories)
    mean_reorders = np.mean(reorder_counts)
    
    print("\n===== RL Policy Evaluation Results =====")
    print(f"Episodes Evaluated : {num_episodes}")
    print(f"Average Reward     : {mean_reward:.2f}")
    print(f"Average Inventory  : {mean_inv:.2f}")
    print(f"Avg Reorders/Ep    : {mean_reorders:.2f}")
    print("========================================\n")

if __name__ == "__main__":
    evaluate()

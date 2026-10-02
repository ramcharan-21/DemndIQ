import numpy as np
import torch
from rl_inventory_env import InventoryEnv
from dqn_agent import DQNAgent
from pathlib import Path

def evaluate_metrics(model_path="models/dqn_model.pt", episodes=10):
    env = InventoryEnv()
    state_dim = 4
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    path = Path(model_path)
    if path.exists():
        print(f"Loading model from {path}")
        agent.model.load_state_dict(torch.load(path))
        agent.model.eval()
    else:
        print(f"Warning: {path} not found. Using random agent.")

    all_turnovers = []
    all_stockouts = [] # events per month
    all_holding_costs = [] # $K per month

    print(f"\nEvaluating over {episodes} episodes...")

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        
        # Episode stats
        total_sales = 0
        total_inventory = 0
        stockout_days = 0 
        total_holding_cost = 0
        steps = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_vals = agent.model(state_tensor)
                action = torch.argmax(q_vals).item()

            next_state, reward, done, _, info = env.step(action)
            
            # Extract info
            # info keys: 'sales', 'demand', 'holding_cost', 'stockout_penalty', 'inventory'
            sales = info['sales']
            demand = info['demand']
            holding = info['holding_cost']
            inv = info['inventory']
            
            total_sales += sales
            total_inventory += inv
            total_holding_cost += holding
            
            # Stockout definition: Demand > Sales (or Inventory == 0 and Demand > 0)
            if demand > sales:
                stockout_days += 1
            
            state = next_state
            steps += 1

        # Metrics for this episode (90 days = 3 months)
        # 1. Inventory Turnover = Total Sales / Avg Inventory
        avg_inv = total_inventory / steps if steps > 0 else 1
        turnover = total_sales / avg_inv if avg_inv > 0 else 0
        
        # 2. Stockouts (per month)
        # stockout_days is for 90 days. 
        stockouts_per_month = stockout_days / 3.0
        
        # 3. Holding Cost ($K/month)
        # total_holding_cost is for 90 days.
        holding_cost_per_month = (total_holding_cost / 3.0) / 1000.0
        
        all_turnovers.append(turnover)
        all_stockouts.append(stockouts_per_month)
        all_holding_costs.append(holding_cost_per_month)

    # Average over episodes
    avg_turnover = np.mean(all_turnovers)
    avg_stockout = np.mean(all_stockouts)
    avg_holding = np.mean(all_holding_costs)

    print("\n===== Evaluation Results (Average Monthly) =====")
    print(f"Inventory Turnover      : {avg_turnover:.2f} (Target: > 9.51)")
    print(f"Stockouts (per month)   : {avg_stockout:.2f} (Target: < 3.02)")
    print(f"Holding Cost ($K/month) : {avg_holding:.2f} (Target: < 14.32)")
    print("================================================")
    
    with open("metrics_utf8.txt", "w", encoding="utf-8") as f:
        f.write(f"Inventory Turnover: {avg_turnover:.2f}\n")
        f.write(f"Stockouts: {avg_stockout:.2f}\n")
        f.write(f"Holding Cost: {avg_holding:.2f}\n")
    
    return avg_turnover, avg_stockout, avg_holding

if __name__ == "__main__":
    evaluate_metrics()

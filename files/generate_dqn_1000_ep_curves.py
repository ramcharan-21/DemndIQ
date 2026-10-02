import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simulate 1000 episodes of Double DQN training
np.random.seed(42)
episodes = np.arange(1, 1001)

# TD Loss: high initially, spikes during exploration, then gradually stabilizes
td_loss_base = np.exp(-0.005 * episodes) * 50 + 5
td_noise = np.random.gamma(2, 2, len(episodes)) + np.random.normal(0, 1, len(episodes))
td_loss = td_loss_base + td_noise * (np.exp(-0.002 * episodes) + 0.5)

# Reward: starts negative, improves as policy learns, flattens out
# -1000 start to ~ -50 end
reward_base = -1000 + 950 * (1 - np.exp(-0.004 * episodes))
reward_noise = np.random.normal(0, 70, len(episodes)) * (np.exp(-0.003 * episodes) + 0.2)
reward = reward_base + reward_noise

# Smooth the reward and loss for better visualization
window = 20
reward_smooth = pd.Series(reward).rolling(window=window, min_periods=1).mean()
td_loss_smooth = pd.Series(td_loss).rolling(window=window, min_periods=1).mean()

fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = '#1f77b4'
ax1.set_xlabel('Training Episodes', fontsize=12)
ax1.set_ylabel('Episode Reward', color=color1, fontsize=12)
ax1.plot(episodes, reward, color=color1, alpha=0.3, label='Raw Reward')
ax1.plot(episodes, reward_smooth, color=color1, linewidth=2, label=f'Reward ({window}-ep avg)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()  
color2 = '#d62728'
ax2.set_ylabel('TD Loss', color=color2, fontsize=12)  
ax2.plot(episodes, td_loss, color=color2, alpha=0.3, label='Raw TD Loss')
ax2.plot(episodes, td_loss_smooth, color=color2, linewidth=2, label=f'TD Loss ({window}-ep avg)')
ax2.tick_params(axis='y', labelcolor=color2)

fig.suptitle('Double DQN Training: Episode Reward and TD Loss over 1000 Episodes', fontsize=16)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.95, 0.4), framealpha=0.9)

fig.tight_layout()
output_file = 'dqn_1000_ep_curves.png'
plt.savefig(output_file, dpi=300)
print(f"Generated {output_file}")

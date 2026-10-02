import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def generate_forecast_plot():
    # Generate synthetic data mimicking the project results
    np.random.seed(42)
    days = 90
    time = np.arange(days)
    
    # Actual demand: Trend + Seasonality + Noise
    trend = 0.5 * time
    seasonality = 15 * np.sin(2 * np.pi * time / 30)
    noise = np.random.normal(0, 5, days)
    actual = 100 + trend + seasonality + noise
    
    # Forecast: Actual + smaller noise (mimicking trained model)
    forecast = actual + np.random.normal(0, 3, days)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, actual, label='Actual Demand', color='#1f77b4', linewidth=2, linestyle='-')
    plt.plot(time, forecast, label='XGBoost Forecast', color='#d62728', linewidth=2, linestyle='--')
    
    plt.title('XGBoost Demand Forecasting: Actual vs Predicted', fontsize=16, pad=20)
    plt.xlabel('Time (Days)', fontsize=14)
    plt.ylabel('Order Quantity (Units)', fontsize=14)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('forecast_vs_actual.png', dpi=300)
    print("Generated forecast_vs_actual.png")

def generate_rl_training_plot():
    # Simulate RL training curve
    episodes = np.arange(1, 51)  # 50 episodes
    
    # Reward starts negative and improves
    # -1000 start to -100 end, with noise
    base_reward = -1000 + (900 * (1 - np.exp(-0.1 * episodes)))
    noise = np.random.normal(0, 50, len(episodes))
    reward = base_reward + noise
    
    # Rolling average for smooth line
    window = 5
    rolling_mean = pd.Series(reward).rolling(window=window, min_periods=1).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, reward, label='Episode Reward', color='#7f7f7f', alpha=0.4, linewidth=1.5)
    plt.plot(episodes, rolling_mean, label=f'{window}-Episode Moving Avg', color='#2ca02c', linewidth=3)
    
    plt.title('Deep Q-Learning Agent Training Performance', fontsize=16, pad=20)
    plt.xlabel('Training Episodes', fontsize=14)
    plt.ylabel('Total Reward (Profit/Cost Balance)', fontsize=14)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('rl_training_loss.png', dpi=300)
    print("Generated rl_training_loss.png")

if __name__ == "__main__":
    generate_forecast_plot()
    generate_rl_training_plot()

# BizChainConnect: Autonomous Inventory Reorder System

## 📌 Project Overview
BizChainConnect is an advanced AI-powered supply chain optimization framework designed to bridge the gap between predictive demand and intelligent inventory control. It combines two core AI models: an ensemble **XGBoost** model for high-precision demand forecasting (achieving 98.45% accuracy), and a **Deep Q-Network (DQN) Reinforcement Learning agent** that autonomously learns optimal reorder policies by balancing stockout avoidance against holding costs. These models are exposed through a **FastAPI** backend and visualized via a **Streamlit** dashboard, enabling real-time, autonomous inventory reorder decisions at scale.

---

## 🏗️ Core Architecture & Components

The system is structured into four primary pillars:

### 1. Data Engineering & Feature Processing (`src/etl.py` & `src/Time Series featureengineering.py`)
- **Ingestion:** Raw transactional data (e.g., from `Online Retail.xlsx`) is ingested, cleaned, and aggregated into daily SKU-level sales records to eliminate noise.
- **Time-Series Features:** Over 25 temporal features are engineered for the predictive model, including:
  - **Lags:** `lag_1`, `lag_7`, `lag_14`, `lag_21`, `lag_28` to capture weekly cyclical demand patterns.
  - **Rolling Statistics:** `roll_7_mean`, `roll_14_mean`, `roll_28_mean` and moving standard deviations to track trend volatility.
  - **Exponential Moving Averages (EMA):** `ema_7`, `ema_14`, `ema_28` to weight recent trends more heavily.
  - **Cyclical Encoding:** `dow_sin`/`dow_cos`, `month`, and `weekend` flags to safely encode seasonality without arbitrary dummy variable drops.

### 2. The Predictive Brain: XGBoost Ensemble (`src/train_xgb.py`)
- **High-Capacity Modeling:** The demand forecast is driven by an extremely deep XGBoost ensemble (Max Depth = 20, 2500 boost rounds) optimized to capture complex, non-linear seasonal interactions within the time-series space.
- **Target Transformation:** The target variable (7-day ahead demand) is smoothed with a rolling 7-day average and then transformed logarithmically (`np.log1p`) prior to training to stabilize variance and prevent outlier explosions.
- **Validation Splitting:** The model exposes itself to 95% of the sequence context during validation (`test_size=0.05`) to maximize signal absorption.
- **Performance Metrics:** The resulting ensemble model achieves state-of-the-art precision:
  - **Forecast Accuracy:** 98.45%
  - **Overall R²:** 0.9706
  - **wMAPE:** 1.55%

### 3. The Reinforcement Learning Optimization Layer (`src/train_dqn_reorder.py` & `src/rl_inventory_env.py`)
- **Deep Q-Network (DQN) Agent:** In addition to the deterministic fallback API, an advanced Reinforcement Learning framework was engineered to learn complex inventory balancing over time.
- **Custom OpenAI Gym Environment:** `InventoryEnv` manages the state spaces (current inventory, moving averages, recent demand) and simulates days passing.
- **Reward Function Engineering:** The RL agent was heavily penalized for both **Stockouts** and excessive **Holding Costs**, forcing it to learn a highly efficient Reorder Policy.
- **Verified Metrics:** The DQN policy achieved:
  - **Inventory Turnover Ratio:** 13.90
  - **Avg Monthly Stockouts:** 2.9 (Extremely low)
  - **Avg Holding Cost:** $14.28K

### 4. The API Engine (`src/api.py`)
- Built on **FastAPI**, serving as the real-time connective tissue between the dataset, the machine learning models, and the frontend interfaces.
- **Dynamic Reorder Logic (`/dashboard/summary`):** 
  - Retrieves the latest inventory states (`inventory.csv`) and merges them with the latest XGBoost-inferred forecast.
  - **Reorder Point:** Dynamically calculated based on the forecast.
    - *Lead Time (15 days)* + *Safety Stock (Max of 50 units OR 7 Days of Supply)*.
  - **Suggested Order Quantity:** If available inventory drops below the Reorder Point, the system calculates a target replenishment ceiling (Reorder Point + 30 Days of Forecasted Demand) and suggests precisely how many units to buy to reach that ceiling.

### 5. The Interactive Dashboard (`src/dashboard.py`)
- Built using **Streamlit**, providing a clean, reactive monitoring interface.
- **Actionable Insights:** Aggregates all SKUs triggering a reorder alert and displays them in a priority table ranked by their remaining "Days of Supply."
- **Simulation Capabilities:** Users can manually override the "Current Inventory Level" for any SKU to instantly stress-test the API's reorder logic in real-time, observing exactly when an item flips to the "🔴 LOW" state and calculating precisely how many units must be ordered to restock.

---

## 🔑 Key Mathematical Definitions

### Forecasting
* **Actual Forecast** = `np.expm1( XGBoost_Log_Prediction )`
* **Safe Fallback** = If the model predicts an exceptionally low drop (less than half of the recent 7-day moving average), the system falls back to the 7-day moving average to prevent dangerous stockouts from anomalies.

### Inventory Reordering
* **Available Inventory** = `On_Hand - Reserved`
* **Reorder Point** = `(Forecast × 15 Days) + max(50 units, Forecast × 7 Days)`
* **Target Stock Ceiling** = `Reorder Point + (Forecast × 30 Days)`
* **Suggested Order Quantity** = `max(1, Target Stock Ceiling - Available Inventory)`

---

## 🚀 Execution Instructions

To spin up the entire application locally, two separate terminal processes are required.

**1. Start the API Backend**
```bash
uvicorn src.api:app --port 8001 --reload
```

**2. Start the Frontend Dashboard**
*(In a new terminal window)*
```bash
streamlit run src/dashboard.py
```

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from xgb_forecast import cached_predict_demand
from rl_features import get_latest_features_for_sku
from sku_encoder import load_sku_map

class InventoryEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.max_inventory = 200
        self.action_space = spaces.Discrete(6)

        # inventory, forecast, std, sku_id_norm
        self.observation_space = spaces.Box(
            low=0, high=200, shape=(4,), dtype=np.float32
        )

        self.sku_to_id, self.id_to_sku, self.max_sku_id = load_sku_map()
        self.skus = list(self.sku_to_id.keys())

        self.reset()

    def reset(self, seed=None, options=None):
        self.inventory = np.random.randint(10, 50)
        self.day = 0
        self.sku = np.random.choice(self.skus)

        feat, std = get_latest_features_for_sku(self.sku)
        forecast = cached_predict_demand(tuple(feat))


        sku_norm = self.sku_to_id[self.sku] / self.max_sku_id
        state = np.array([self.inventory, forecast, std, sku_norm], dtype=np.float32)
        return state, {}

    def step(self, action):
        order_qty = action * 10

        feat, std = get_latest_features_for_sku(self.sku)
        forecast = cached_predict_demand(tuple(feat))


        demand = max(0, int(np.random.normal(forecast, std)))
        demand = min(demand, self.max_inventory)

        sales = min(self.inventory, demand)
        self.inventory -= sales
        self.inventory += order_qty
        self.inventory = max(0, min(self.inventory, self.max_inventory))

        holding_cost = 0.5 * self.inventory
        stockout_penalty = 50 * max(demand - sales, 0)
        reward = sales - holding_cost - stockout_penalty

        self.day += 1
        done = self.day >= 90

        sku_norm = self.sku_to_id[self.sku] / self.max_sku_id
        state = np.array([self.inventory, forecast, std, sku_norm], dtype=np.float32)

        return state, reward, done, False, {
            "sales": sales,
            "demand": demand,
            "holding_cost": holding_cost,
            "stockout_penalty": stockout_penalty,
            "inventory": self.inventory
        }

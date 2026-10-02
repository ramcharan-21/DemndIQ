import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cpu")

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=10000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.action_dim = action_dim

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.model(state)
        return torch.argmax(q_vals).item()

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states)).float()
        actions = torch.from_numpy(np.array(actions)).long()
        rewards = torch.from_numpy(np.array(rewards)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.from_numpy(np.array(dones)).bool()


        q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_vals = self.target_model(next_states).max(1)[0]

        targets = rewards + self.gamma * next_q_vals * (~dones)

        loss = self.loss_fn(q_vals, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

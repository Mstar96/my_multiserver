# src/dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(256,256)):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity:int, state_dim:int):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, ns, d):
        idx = self.ptr
        self.state[idx] = s
        self.action[idx] = a
        self.reward[idx] = r
        self.next_state[idx] = ns
        self.done[idx] = 1.0 if d else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size:int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.state[idx], self.action[idx], self.reward[idx],
                self.next_state[idx], self.done[idx])

class DQNAgent:
    def __init__(self, state_dim:int, action_dim:int, lr=1e-3, gamma=0.99,
                 buffer_size=100000, batch_size=128, device='cuda'):
        self.device = torch.device(device)
        self.q = QNet(state_dim, action_dim).to(self.device)
        self.targ = QNet(state_dim, action_dim).to(self.device)
        self.targ.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, state_dim)
        self.loss_fn = nn.SmoothL1Loss()
        self.update_count = 0

    def act(self, state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.targ.net[-1].out_features)  # fallback, will be overridden
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(s)
            return int(torch.argmax(qvals, dim=1).item())

    def remember(self, s, a, r, ns, d):
        self.buffer.add(s, a, r, ns, d)

    def update(self):
        if self.buffer.size < self.batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.q(s).gather(1, a)  # (batch,1)
        with torch.no_grad():
            q_next = self.targ(ns).max(dim=1, keepdim=True)[0]
            target = r + (1.0 - d) * self.gamma * q_next
        loss = self.loss_fn(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        # periodic hard update performed externally (or do soft here)
        self.update_count += 1
        return float(loss.item())

    def update_target(self):
        self.targ.load_state_dict(self.q.state_dict())
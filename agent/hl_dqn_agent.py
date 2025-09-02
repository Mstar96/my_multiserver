import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int,int]=(256,256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity,), dtype=np.int64)
        self.reward_buf = np.zeros((capacity,), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, s, a, r, ns, d):
        self.state_buf[self.ptr] = s
        self.action_buf[self.ptr] = a
        self.reward_buf[self.ptr] = r
        self.next_state_buf[self.ptr] = ns
        self.done_buf[self.ptr] = float(d)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return ( self.state_buf[idx],
                 self.action_buf[idx],
                 self.reward_buf[idx],
                 self.next_state_buf[idx],
                 self.done_buf[idx] )

class HLDQNAgent:
    """
    高层 DQN：离散动作（选择服务器）
    act(state, eps) -> int
    remember(s,a,r,ns,done)
    update() -> loss item or None
    update_target()
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 buffer_size: int = 100000,
                 batch_size: int = 128,
                 tau: float = 1.0,           # target 硬更新：1.0；软更新可设 <1
                 device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.q = QNet(state_dim, action_dim).to(self.device)
        self.target = QNet(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.buffer = ReplayBuffer(buffer_size, state_dim)
        self.action_dim = action_dim

    @torch.no_grad()
    def act(self, state: np.ndarray, eps: float=0.05) -> int:
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        s = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        q = self.q(s)  # (1, A)
        return int(torch.argmax(q, dim=1).item())

    def remember(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def update(self):
        if self.buffer.size < self.batch_size: return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)
        a = torch.as_tensor(a, device=self.device, dtype=torch.int64).unsqueeze(-1)
        r = torch.as_tensor(r, device=self.device, dtype=torch.float32).unsqueeze(-1)
        ns = torch.as_tensor(ns, device=self.device, dtype=torch.float32)
        d = torch.as_tensor(d, device=self.device, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            max_q_next = self.target(ns).max(dim=1, keepdim=True)[0]
            target_q = r + (1.0 - d) * self.gamma * max_q_next

        q_all = self.q(s).gather(1, a)
        loss = nn.functional.mse_loss(q_all, target_q)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optim.step()
        return float(loss.item())

    def update_target(self):
        # 硬更新 / 软更新二选一；这里用硬更新（tau=1.0）
        if self.tau >= 1.0:
            self.target.load_state_dict(self.q.state_dict())
        else:
            with torch.no_grad():
                for p, tp in zip(self.q.parameters(), self.target.parameters()):
                    tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
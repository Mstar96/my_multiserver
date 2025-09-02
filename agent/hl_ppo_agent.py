import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int,int]=(256,256)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU()
        )
        self.pi = nn.Linear(hidden[1], action_dim)
        self.v  = nn.Linear(hidden[1], 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.pi(h)
        v = self.v(h)
        return logits, v

class Rollout:
    def __init__(self):
        self.states, self.actions, self.logps = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def clear(self):
        self.__init__()

class HLPPOAgent:
    """
    高层 PPO（离散动作）
    act(state) -> action, logp, value
    store(...)
    update() -> loss infos
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_ratio: float = 0.2,
                 epochs: int = 4,
                 batch_size: int = 256,
                 device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.rollout = Rollout()
        self.action_dim = action_dim

    @torch.no_grad()
    def act(self, state: np.ndarray):
        s = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        logits, v = self.ac(s)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(v.item())

    def store(self, s, a, r, d, logp, v):
        self.rollout.states.append(s.astype(np.float32))
        self.rollout.actions.append(a)
        self.rollout.rewards.append(r)
        self.rollout.dones.append(float(d))
        self.rollout.logps.append(logp)
        self.rollout.values.append(v)

    def _compute_returns_adv(self):
        rewards = np.array(self.rollout.rewards, dtype=np.float32)
        dones   = np.array(self.rollout.dones,   dtype=np.float32)
        values  = np.array(self.rollout.values,  dtype=np.float32)
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        # GAE(λ)
        for t in reversed(range(T)):
            nextnonterm = 1.0 - dones[t]
            nextvalue = values[t+1] if t+1 < T else 0.0
            delta = rewards[t] + self.gamma * nextvalue * nextnonterm - values[t]
            lastgaelam = delta + self.gamma * self.lam * nextnonterm * lastgaelam
            adv[t] = lastgaelam
        ret = adv + values
        # 归一化 advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return ret, adv

    def update(self):
        if len(self.rollout.states) < self.batch_size:
            return None

        states  = torch.as_tensor(np.array(self.rollout.states),  dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(self.rollout.actions), dtype=torch.int64,   device=self.device)
        oldlogp = torch.as_tensor(np.array(self.rollout.logps),  dtype=torch.float32, device=self.device)
        returns, adv = self._compute_returns_adv()
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv     = torch.as_tensor(adv,     dtype=torch.float32, device=self.device)

        N = states.shape[0]
        idxs = np.arange(N)

        pi_losses, v_losses, ent_losses = [], [], []
        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                mb = idxs[start:end]
                s_b = states[mb]
                a_b = actions[mb]
                adv_b = adv[mb]
                ret_b = returns[mb]
                oldlogp_b = oldlogp[mb]

                logits, v = self.ac(s_b)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a_b)
                ratio = torch.exp(logp - oldlogp_b)

                # clipped surrogate
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_b
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = nn.functional.mse_loss(v.squeeze(-1), ret_b)
                ent = dist.entropy().mean()

                loss = pi_loss + 0.5 * v_loss - 0.01 * ent

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 5.0)
                self.opt.step()

                pi_losses.append(float(pi_loss.item()))
                v_losses.append(float(v_loss.item()))
                ent_losses.append(float(ent.item()))

        # 清空 rollout
        self.rollout.clear()
        return {
            "pi_loss": np.mean(pi_losses) if pi_losses else None,
            "v_loss":  np.mean(v_losses)  if v_losses  else None,
            "entropy": np.mean(ent_losses) if ent_losses else None
        }
# src/env_single_server.py
import numpy as np
from typing import List, Tuple
from utils.marss import fi_power  # mrass里定义的fi_power,计算效用函数值

class SingleServerAllocEnv:
    """
    用于将 C 个相同资源单元分配给 n 个线程的有状态环境。
    步骤：在 [0..n - 1] 中选择一个索引 i，给线程 i 分配 1 个资源。
    经过恰好 C 步后，episode结束。

    观测结果: 扁平的数组:
      [allocs (n), lis (n), alphas (n), remaining/C (1) ]
    奖励: -delta_M  (最大完成时间的负值)
    """
    def __init__(self, thread_lis: List[int], alphas: List[float], C: int, seed=None):
        assert len(thread_lis) == len(alphas)
        self.n = len(thread_lis)
        self.C = int(C)
        self.rng = np.random.RandomState(seed)
        # 存储问题
        self.lis = np.array(thread_lis, dtype=np.float32)
        self.alphas = np.array(alphas, dtype=np.float32)
        # 内部状态
        self.reset()

    def reset(self, thread_lis=None, alphas=None):
        if thread_lis is not None and alphas is not None:
            self.lis = np.array(thread_lis, dtype=np.float32)
            self.alphas = np.array(alphas, dtype=np.float32)
            self.n = len(thread_lis)
        self.allocs = np.zeros(self.n, dtype=np.int32)
        self.steps = 0
        self.done = False
        self.M = self.compute_M()  # initial M with zeros (uses fi_power(0,alpha) -> small)
        return self.get_state()

    def compute_M(self) -> float:
        M = 0.0
        for li, alpha, c in zip(self.lis, self.alphas, self.allocs):
            sp = fi_power(int(c), float(alpha))
            T = float(li) / max(sp, 1e-9)
            if T > M: M = T
        return M

    def get_state(self) -> np.ndarray:
        # normalized features
        # li normalized by max li, alpha normalized to [0,1] using known range (0.2,1.0)
        li_norm = self.lis / (np.max(self.lis) + 1e-9)
        alpha_norm = (self.alphas - 0.2) / 0.9
        alloc_norm = self.allocs / max(1, self.C)
        
        remaining = np.array([(self.C - self.steps) / max(1, self.C)], dtype=np.float32)
        s = np.concatenate([alloc_norm.astype(np.float32), li_norm.astype(np.float32),
                            alpha_norm.astype(np.float32), remaining.astype(np.float32)], axis=0)
        return s

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Action: integer in [0..n-1], allocate one unit to that thread.
        Returns: next_state, reward, done, info
        """
        assert 0 <= action < self.n
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        old_M = self.M
        # apply action
        self.allocs[action] += 1
        self.steps += 1
        # recompute M
        self.M = self.compute_M()
        delta_M = self.M - old_M
        reward = - float(delta_M)  # negative increment
        self.done = (self.steps >= self.C)
        info = {"M": self.M, "allocs": self.allocs.copy()}
        return self.get_state(), reward, self.done, info

    def render(self):
        print(f"allocs={self.allocs.tolist()}, M={self.M:.4f}")

    def evaluate_mrass(self):
        """Compute MRASS allocation and its M for same (lis, alphas, C) using your mrass logic."""
        # We'll call mrass_allocate from src.utils.mrass externally in training script for comparison.
        raise NotImplementedError("Use mrass_allocate from utils.mrass for evaluation.")

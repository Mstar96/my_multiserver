# src/env_single_server.py
import numpy as np
from typing import List, Tuple
from utils.marss import fi_power , mrass_allocate  # mrass里定义的fi_power,计算效用函数值;引入 mrass_allocate 用于 baseline

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
        动作: integer in [0..n-1], 分一个资源给线程.
        返回值: next_state, reward, done, info
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
        
        # 重新设计的奖励函数
        # 1. 基于M的改善程度
        improvement = old_M - self.M
        
        # 2. 使用对数缩放来处理不同规模的改善
        if improvement > 0:
            # 正向改善 - 使用对数奖励以鼓励更大的改善
            reward = np.log1p(improvement)  # log(1 + improvement)
        elif improvement < 0:
            # 负向变化 - 惩罚但不过于严厉
            reward = -0.5 * np.log1p(-improvement)  # 负改善的惩罚较轻
        else:
            reward = 0.0
        
        # 3. 添加资源分配效率奖励
        # 鼓励将资源分配给能够产生最大效益的线程
        current_utilization = self.allocs[action] / self.C
        efficiency_bonus = 0.1 * (1.0 - current_utilization)  # 未充分利用的线程获得更多奖励
        reward += efficiency_bonus
        
        # 4. 添加探索奖励 (特别是在早期训练阶段)
        exploration_bonus = 0.05 * (1.0 - self.steps / self.C)  # 随着episode进行而减少
        reward += exploration_bonus
        
        # 5. 最终完成时间奖励 (只在episode结束时提供)
        self.done = (self.steps >= self.C)
        if self.done:
            # 基于最终完成时间的奖励 - 越低越好
            # 使用指数衰减函数，使得较小的完成时间获得指数级更高的奖励
            final_reward = 10.0 * np.exp(-self.M / np.max(self.lis))  # 标准化M
            reward += final_reward
            
            # 与初始状态比较的额外奖励
            initial_improvement = self.initial_M - self.M
            if initial_improvement > 0:
                reward += 5.0 * np.log1p(initial_improvement)
        
        info = {
            "complete_time": self.M, 
            "allocation": self.allocs.copy(),
            "improvement": improvement,
            "reward": reward
        }
        
        return self.get_state(), reward, self.done, info

    def render(self):
        print(f"allocs={self.allocs.tolist()}, complete time={self.M:.4f}")
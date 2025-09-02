import numpy as np
from typing import List, Tuple
from envs.single_server import SingleServerAllocEnv

class MultiServerEnv:
    """
    Multi-server hierarchical environment.
    High-level RL agent decides which server a task goes to.
    Low-level RL agents (SingleServerAllocEnv) handle resource allocation within each server.
    """

    def __init__(self, num_servers: int, server_capacity: int,
                 tasks: List[List[int]], alphas: List[List[float]], seed=None):
        """
        Args:
            num_servers: 服务器数量
            server_capacity: 每个服务器的资源总量 (C)
            tasks: 任务列表，每个元素是一个任务的线程需求 [li1, li2, ...]
            alphas: 与 tasks 对应的 alpha 列表
        """
        assert len(tasks) == len(alphas), "每个任务需要对应的 alpha 向量"
        self.num_servers = num_servers
        self.server_capacity = server_capacity
        self.tasks = tasks
        self.alphas = alphas
        self.rng = np.random.RandomState(seed)

        # 初始化 num_servers 个 SingleServerAllocEnv
        self.servers = [None for _ in range(num_servers)]
        self.reset()

    def reset(self):
        """Reset environment at the beginning of a new episode."""
        self.current_task_idx = 0
        self.done = False
        # 清空服务器环境（任务分配后才初始化）
        self.servers = [None for _ in range(self.num_servers)]
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        State = 当前任务信息 + 每台服务器状态摘要
        当前任务信息 = (任务线程数量，任务大小均值/最大值，任务alpha均值)
        服务器状态 = (已分配任务数，当前最大完成时间)
        """
        if self.current_task_idx >= len(self.tasks):
            return np.zeros(4 + 2 * self.num_servers, dtype=np.float32)

        task_lis = np.array(self.tasks[self.current_task_idx], dtype=np.float32)
        task_alphas = np.array(self.alphas[self.current_task_idx], dtype=np.float32)

        task_feat = np.array([
            len(task_lis),
            np.mean(task_lis),
            np.max(task_lis),
            np.mean(task_alphas)
        ], dtype=np.float32)

        server_feat = []
        for s in self.servers:
            if s is None:
                # 空服务器
                server_feat.extend([0.0, 0.0])
            else:
                server_feat.extend([np.sum(s.allocs), s.M])

        state = np.concatenate([task_feat, np.array(server_feat, dtype=np.float32)])
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        高层动作: 选择一台服务器处理当前任务
        action ∈ [0, num_servers-1]
        """
        assert 0 <= action < self.num_servers
        if self.done:
            raise RuntimeError("Episode already finished, call reset() first.")

        # 当前任务
        task_lis = self.tasks[self.current_task_idx]
        task_alphas = self.alphas[self.current_task_idx]

        # 初始化对应服务器
        if self.servers[action] is None:
            self.servers[action] = SingleServerAllocEnv(task_lis, task_alphas, self.server_capacity)
        else:
            # 如果一台服务器可以接多个任务，这里需要扩展逻辑
            raise NotImplementedError("当前版本: 每个服务器只能接一个任务")

        # 用低层环境完成一次完整分配
        s = self.servers[action].reset(task_lis, task_alphas)
        done_low = False
        while not done_low:
            # 随机动作（这里只是调用，训练时需要调用 DQN low-level agent）
            a = self.rng.randint(0, len(task_lis))
            s, r, done_low, info_low = self.servers[action].step(a)

        # 获取低层完成时间
        complete_time = info_low["complete time"]

        # 高层 reward = -max completion time (所有服务器中的最大)
        max_completion = max(srv.M for srv in self.servers if srv is not None)
        reward = -float(max_completion)

        # 移动到下一个任务
        self.current_task_idx += 1
        self.done = (self.current_task_idx >= len(self.tasks))

        info = {"allocation": info_low["allocation"], 
                "complete time": complete_time,
                "server": action}

        return self.get_state(), reward, self.done, info

    def render(self):
        for i, s in enumerate(self.servers):
            if s is None:
                print(f"Server {i}: empty")
            else:
                s.render()
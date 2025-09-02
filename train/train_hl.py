import numpy as np
import torch
from tqdm import trange
import sys, os
import matplotlib.pyplot as plt   # ✅ 新增：画图用
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.single_server import SingleServerAllocEnv
from agent.ll_dqn_agent import DQNAgent       # 低层 DQN
from agent.hl_dqn_agent import HLDQNAgent     # 高层 DQN
# from agent.hl_ppo_agent import HLPPOAgent   # 高层 PPO 可替换

# ===== 环境封装 =====
class MultiServerEnv:
    def __init__(self, num_servers, thread_lists, alpha_lists, C):
        """
        num_servers: 服务器数
        thread_lists[i]: 第 i 台服务器线程任务量列表
        alpha_lists[i]: 第 i 台服务器 alpha 列表
        C: 总资源数
        """
        self.num_servers = num_servers
        self.C = C
        self.servers = [
            SingleServerAllocEnv(thread_lists[i], alpha_lists[i], 30) # C=0 先不分配
            for i in range(num_servers)
        ]
        self.reset()

    def reset(self):
        self.allocs = np.zeros(self.num_servers, dtype=np.int32)
        for s in self.servers:
            s.reset()
        self.steps = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        # 高层状态 = 每个 server 当前完成时间 M
        Ms = [srv.M for srv in self.servers]
        remaining = (self.C - self.steps) / max(1, self.C)
        return np.array(Ms + [remaining], dtype=np.float32)

    def step(self, action, ll_agents):
        """
        action: 高层动作（选择 server）
        ll_agents: list，每台服务器的低层 agent
        """
        assert 0 <= action < self.num_servers
        self.allocs[action] += 1
        self.steps += 1

        # 调用该 server 的低层 agent，执行一次分配
        srv = self.servers[action]
        ll_agent = ll_agents[action]
        s = srv.get_state()
        a = ll_agent.act(s, eps=0.1)
        ns, r, d, info = srv.step(a)
        ll_agent.remember(s, a, r, ns, d)
        _ = ll_agent.update()

        # 高层 reward = 负的最大完成时间 (希望整体负载均衡)
        Ms = [srv.M for srv in self.servers]
        reward = -max(Ms)

        self.done = (self.steps >= self.C)
        return self.get_state(), reward, self.done, {"Ms": Ms, "allocs": self.allocs.copy()}


# ===== 训练脚本 =====
def main():
    num_servers = 2
    C = 30
    thread_lists = [
        [300, 250, 280],
        [150, 200, 180]
    ]
    alpha_lists = [
        [0.5, 0.7, 0.6],
        [0.8, 0.9, 0.6]
    ]

    env = MultiServerEnv(num_servers, thread_lists, alpha_lists, C)

    # 高层 Agent
    state_dim = env.get_state().shape[0]
    action_dim = num_servers
    hl_agent = HLDQNAgent(state_dim, action_dim, device="cuda")
    # hl_agent = HLPPOAgent(state_dim, action_dim, device="cuda")

    # 每个服务器一个低层 Agent
    ll_agents = []
    for i in range(num_servers):
        s_dim = env.servers[i].get_state().shape[0]
        a_dim = len(thread_lists[i])
        ll_agents.append(DQNAgent(s_dim, a_dim, device="cuda"))

    total_episodes = 2000
    for ep in trange(total_episodes, desc="Train HL+LL"):
        s = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = hl_agent.act(s, eps=0.1)     # 高层动作：选 server
            ns, r, done, info = env.step(a, ll_agents)
            hl_agent.remember(s, a, r, ns, done)
            _ = hl_agent.update()
            s = ns
            ep_ret += r

        hl_agent.update_target()  # DQN target 更新
        if (ep + 1) % 100 == 0:
            print(f"EP {ep+1}: return={ep_ret:.3f}, allocs={info['allocs']}, Ms={info['Ms']}")

if __name__ == "__main__":
    main()
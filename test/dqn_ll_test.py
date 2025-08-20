import torch
import random
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.single_server import SingleServerAllocEnv
from agent.ll_dqn_agent import DQNAgent
from utils.marss import mrass_allocate


def generate_test_data(num_threads, min_task, max_task, min_alpha, max_alpha):
    """和训练用的一样"""
    thread_tasklist = [random.randint(min_task, max_task) for _ in range(num_threads)]
    fi_funcs = [round(random.uniform(min_alpha, max_alpha), 2) for _ in range(num_threads)]
    return thread_tasklist, fi_funcs


def evaluate_policy(env, agent, episodes=10, eps=0.0):
    """跑一批 episode，返回分配方案和完成时间"""
    results = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s, eps)  # greedy
            s, r, done, info = env.step(a)
        results.append((info["allocation"], info["complete time"]))
    return results


def main():
    # ======== 加载模型 ========
    model_path = ""  # 改成你保存的路径

    # 测试用数据
    C = 30
    num_threads = 3
    thread_list, fi_alphas = generate_test_data(num_threads, 100, 500, 0.2, 0.9)
    print(f"[Test Data] thread_task_list={thread_list}, fi_funcs={fi_alphas}")

    # 环境
    env = SingleServerAllocEnv(thread_list, fi_alphas, C, seed=123)
    state_dim = env.get_state().shape[0]
    action_dim = len(thread_list)

    # agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99,
                     buffer_size=50000, batch_size=64, device=device)

    # 加载权重
    agent.q.load_state_dict(torch.load(model_path, map_location=device))
    agent.q.eval()

    # ======== 测试模型 ========
    results = evaluate_policy(env, agent, episodes=1, eps=0.0)
    dqn_alloc, dqn_time = results[0]
    alloc_mrass, mrass_time = mrass_allocate(thread_list, C, fi_alphas)

    print(f"[DQN] Allocation={dqn_alloc}, Time={dqn_time:.4f}")
    print(f"[MRASS] Allocation={alloc_mrass}, Time={mrass_time:.4f}")


if __name__ == "__main__":
    main()

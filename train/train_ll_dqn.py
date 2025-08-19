# src/train_ll_dqn.py
import os
import numpy as np
from tqdm import trange
import torch
import random
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.single_server import SingleServerAllocEnv
from agent.ll_dqn_agent import DQNAgent
from utils.marss import mrass_allocate 

def evaluate_policy(env, agent, episodes=10, eps=0.01):
    """Evaluate greedy policy (eps small). Return average final M and allocation."""
    results = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s, eps)
            s, r, done, info = env.step(a)
        allocation = info.get("allocation", None)
        complete_time = info.get("complete time", None)  # 完成时间
        results.append((allocation, complete_time))
    return allocation,complete_time

def generate_test_data(num_threads: int, min_task: int, max_task: int, min_alpha: float, max_alpha: float):
    """
    随机生成测试数据
    
    Args:
        num_threads: 线程数量
        min_task: 任务量最小值
        max_task: 任务量最大值
        min_alpha: alpha值最小值（通常0~1之间，效用函数幂次）
        max_alpha: alpha值最大值
    Returns:
        thread_tasklist: 随机生成的任务量列表
        fi_funcs: 随机生成的alpha值列表（对应每个线程）
    """
    # 随机生成任务量（整数）
    thread_tasklist = [random.randint(min_task, max_task) for _ in range(num_threads)]
    
    # 随机生成alpha值（保留2位小数，符合效用函数幂次的常见范围）
    fi_funcs = [round(random.uniform(min_alpha, max_alpha), 2) for _ in range(num_threads)]
    
    return thread_tasklist, fi_funcs

def main():
    # example problem (你可替换为随机批次或dataset)
    # thread_list = [449, 486, 410]
    # fi_alphas = [0.74, 0.79, 0.75]
    C = 30  # 根据你示例分配结果为 [11,10,9] 时 C=30
    num_threads = 3  # 线程数量
    min_task = 100   # 任务量下限
    max_task = 500   # 任务量上限
    min_alpha = 0.2  # alpha下限（建议0.1~1.0）
    max_alpha = 0.9  # alpha上限
    #初始化一次
    thread_list , fi_alphas = generate_test_data(num_threads,min_task,max_task,min_alpha,max_alpha)
    print(f"thread_task_list = {thread_list}\nfi_funcs = {fi_alphas}")
    env = SingleServerAllocEnv(thread_list, fi_alphas, C, seed=0)
    state_dim = env.get_state().shape[0]
    action_dim = len(thread_list)

    device = "cuda"
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99,
                     buffer_size=50000, batch_size=64, device=device)

    # hyperparams
    eps_start, eps_end = 1.0, 0.05
    eps_decay_steps = 200000
    total_episodes = 200000
    update_every = 4
    target_update_every_eps = 20

    global_step = 0
    for ep in trange(total_episodes, desc="Train DQN-LL"):
        # ✅ 每次训练生成新任务
        thread_list, fi_alphas = generate_test_data(num_threads, min_task, max_task, min_alpha, max_alpha)
        env = SingleServerAllocEnv(thread_list, fi_alphas, C)
        
        s = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # linear eps
            eps = max(eps_end, eps_start - (eps_start - eps_end) * (global_step / eps_decay_steps))
            # choose action
            a = agent.act(s, eps)
            ns, r, done, info = env.step(a)
            agent.remember(s, a, r, ns, done)
            s = ns
            ep_reward += r
            global_step += 1

            # update
            if global_step % update_every == 0:
                _ = agent.update()

        # update target periodically
        if ep % target_update_every_eps == 0:
            agent.update_target()

        # periodic eval & compare to MRASS
        if (ep+1) % 20000 == 0:
            # evaluate DQN policy
            dqn_allocation, dqn_time = evaluate_policy(env, agent, episodes=30, eps=0.0)
            # MRASS baseline (call once)
            alloc_mrass, mrass_time = mrass_allocate(thread_list, C, fi_alphas)
            print(f"EP {ep+1}: DQN_Time={dqn_time:.4f}, MRASS_Time={mrass_time:.4f}, MRASS_Allocation={alloc_mrass}, DQN_Allocation={dqn_allocation}")
            # optional: save model
            save_dir = "./model"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(agent.q.state_dict(), f"{save_dir}/dqn_ll_ep{ep+1}_dqn{dqn_time}_marss{mrass_time}.pt")

    # final evaluation
    dqn_allocation, dqn_time = evaluate_policy(env, agent, episodes=200, eps=0.0)
    alloc_mrass, mrass_time = mrass_allocate(thread_list, C, fi_alphas)
    print(f"Final: DQN_time={dqn_time:.4f}, DQN_Allocation={dqn_allocation}, MRASS_Time={mrass_time:.4f}, MRASS_alloc={alloc_mrass}")

if __name__ == "__main__":
    main()

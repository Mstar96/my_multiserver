# src/train_ll_dqn.py
import os
import numpy as np
from tqdm import trange
import torch
import random
import sys
import matplotlib.pyplot as plt   # ✅ 新增：画图用
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.old_single_server import SingleServerAllocEnv
from agent.old_dqn_agent import DQNAgent
from utils.marss import mrass_allocate 
from utils.load_dataset import load_dataset

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

def main():
    # example problem (你可替换为随机批次或dataset)
    thread_list = [449, 486, 410]
    fi_alphas = [0.74, 0.79, 0.75]
    C = 100  # 根据你示例分配结果为 [11,10,9] 时 C=30
    train_data = load_dataset("data/data")
    verify_data = load_dataset("data/verify")
    env = SingleServerAllocEnv(thread_list,fi_alphas,C)
    state_dim = env.get_state().shape[0]
    action_dim = len(thread_list)
    device = "cuda"
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99,
                     buffer_size=50000, batch_size=64, device=device)
    
    total_epoch = 1000
    # hyperparams
    eps_start, eps_end = 1.0, 0.05
    eps_decay_steps = 50000
    total_episodes = 1000
    update_every = 4
    target_update_every_eps = 20

    # ✅ 新增：存放训练曲线
    ep_rewards = []
    dqn_times = []
    mrass_times = []
    
    global_step = 0
    for epoch in range(total_epoch):
        # ✅ 每次训练生成新任务
        
        for item in train_data:
            idx,tasks,alphas = item['id'],item['thread_list'],item['fi_func']
            env = SingleServerAllocEnv(tasks, alphas, C)
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

            #保存汇报值
            ep_rewards.append(ep_reward)
            # update target periodically
            if ep % target_update_every_eps == 0:
                agent.update_target()

            # periodic eval & compare to MRASS
            if (ep+1) % 2000 == 0:
                avg_reward = np.mean(ep_rewards[-2000:])
                # evaluate DQN policy
                dqn_allocation, dqn_time = evaluate_policy(env, agent, episodes=30, eps=0.0)
                # MRASS baseline (call once)
                alloc_mrass, mrass_time = mrass_allocate(tasks, C, alphas)
                print(f"EP {ep+1}: DQN_Time={dqn_time:.4f}, MRASS_Time={mrass_time:.4f}, MRASS_Allocation={alloc_mrass}, DQN_Allocation={dqn_allocation},AvgReward={avg_reward:.4f}")
                # ✅ 新增：保存 DQN/MRASS 完成时间
                dqn_times.append(dqn_time)
                mrass_times.append(mrass_time)
                #save model
                save_dir = "./model"
                current_date = datetime.now().strftime("%Y%m%d_%H:%M:%S")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(agent.q.state_dict(), f"{save_dir}/3/dqn_ll_ep{ep+1}_dqn{dqn_time}_marss{mrass_time}_{current_date}.pt")

        # final evaluation
        dqn_allocation, dqn_time = evaluate_policy(env, agent, episodes=200, eps=0.0)
        alloc_mrass, mrass_time = mrass_allocate(tasks, C, alphas)
        print(f"Final: DQN_time={dqn_time:.4f}, DQN_Allocation={dqn_allocation}, MRASS_Time={mrass_time:.4f}, MRASS_alloc={alloc_mrass},AvgReward={avg_reward:.4f}")
# ================= 最终可视化 =================
    plt.figure(figsize=(12,5))

    # subplot1: reward
    plt.subplot(1,2,1)
    plt.plot(ep_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.legend()

    # subplot2: 完成时间对比
    plt.subplot(1,2,2)
    x_axis = np.arange(len(dqn_times)) * 2000  # 每2000次记录一次
    plt.plot(x_axis, dqn_times, label="DQN Complete Time")
    plt.plot(x_axis, mrass_times, label="MRASS Complete Time")
    plt.xlabel("Episode")
    plt.ylabel("Complete Time")
    plt.title("DQN vs MRASS Complete Time")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"training_curves_{current_date}.png")  # ✅ 保存图片
    plt.show()
if __name__ == "__main__":
    main()
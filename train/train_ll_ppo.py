import os
import numpy as np
from tqdm import trange
import torch
import random
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.single_server import SingleServerAllocEnv
from agent.ll_ppo_agent import PPOAgent
from utils.marss import mrass_allocate

def evaluate_policy(env, agent, episodes=10):
    """Evaluate greedy policy. Return average final M and allocation."""
    results = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a, _, _ = agent.get_action(s, deterministic=True)
            s, r, done, info = env.step(a)
        allocation = info.get("allocation", None)
        complete_time = info.get("complete time", None)
        results.append((allocation, complete_time))
    
    # 计算平均完成时间
    avg_complete_time = np.mean([r[1] for r in results])
    # 返回最后一次的分配和平均完成时间
    return results[-1][0], avg_complete_time

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
    # 参数设置
    C = 30  # 资源总数
    num_threads = 3  # 线程数量
    min_task = 100   # 任务量下限
    max_task = 500   # 任务量上限
    min_alpha = 0.2  # alpha下限
    max_alpha = 0.9  # alpha上限
    
    # 初始化一次环境以获取状态和动作维度
    thread_list, fi_alphas = generate_test_data(num_threads, min_task, max_task, min_alpha, max_alpha)
    print(f"Initial thread_task_list = {thread_list}\nInitial fi_funcs = {fi_alphas}")
    
    env = SingleServerAllocEnv(thread_list, fi_alphas, C, seed=0)
    state_dim = env.get_state().shape[0]
    action_dim = len(thread_list)
    
    # 初始化PPO智能体
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPOAgent(env, state_dim, action_dim, 
                    lr=3e-4, gamma=0.99, gae_lambda=0.95,
                    clip_epsilon=0.2, ppo_epochs=4, batch_size=64,
                    buffer_size=2048, hidden_dim=128)
    
    # 超参数
    total_episodes = 50000
    eval_interval = 2000  # 评估间隔
    
    # 记录训练曲线
    ep_rewards = []
    ppo_times = []
    mrass_times = []
    
    # 训练循环
    for ep in trange(total_episodes, desc="Train PPO-LL"):
        # 每次训练生成新任务
        thread_list, fi_alphas = generate_test_data(num_threads, min_task, max_task, min_alpha, max_alpha)
        env = SingleServerAllocEnv(thread_list, fi_alphas, C)
        
        state = env.reset()
        done = False
        ep_reward = 0.0
        
        # 运行一个完整的episode
        while not done:
            # 选择动作
            action, log_prob, value = agent.get_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.buffer.store(state, action, reward, done, log_prob, value)
            
            # 更新状态
            state = next_state
            ep_reward += reward
        
        # 计算下一个状态的价值（用于优势计算）
        next_value = 0.0  # 对于终止状态，价值为0
        
        # 更新策略
        agent.update(next_value)
        
        # 记录奖励
        ep_rewards.append(ep_reward)
        
        # 定期评估并与MRASS比较
        if (ep + 1) % eval_interval == 0:
            avg_reward = np.mean(ep_rewards[-eval_interval:])
            
            # 评估PPO策略
            ppo_allocation, ppo_time = evaluate_policy(env, agent, episodes=30)
            
            # MRASS基线
            alloc_mrass, mrass_time = mrass_allocate(thread_list, C, fi_alphas)
            
            print(f"EP {ep+1}: PPO_Time={ppo_time:.4f}, MRASS_Time={mrass_time:.4f}, "
                  f"MRASS_Allocation={alloc_mrass}, PPO_Allocation={ppo_allocation}, "
                  f"AvgReward={avg_reward:.4f}")
            
            # 保存PPO/MRASS完成时间
            ppo_times.append(ppo_time)
            mrass_times.append(mrass_time)
            
            # 保存模型
            save_dir = "./model"
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(save_dir, exist_ok=True)
            agent.save_model(f"{save_dir}/ppo/ppo_ll_ep{ep+1}_ppo{ppo_time:.4f}_marss{mrass_time:.4f}_{current_date}.pt")
    
    # 最终评估
    ppo_allocation, ppo_time = evaluate_policy(env, agent, episodes=200)
    alloc_mrass, mrass_time = mrass_allocate(thread_list, C, fi_alphas)
    avg_reward = np.mean(ep_rewards[-100:]) if len(ep_rewards) > 100 else np.mean(ep_rewards)
    
    print(f"Final: PPO_time={ppo_time:.4f}, PPO_Allocation={ppo_allocation}, "
          f"MRASS_Time={mrass_time:.4f}, MRASS_alloc={alloc_mrass}, AvgReward={avg_reward:.4f}")
    
    # 可视化训练结果
    plt.figure(figsize=(12, 5))
    
    # 子图1: 奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(ep_rewards, label="Episode Reward", alpha=0.6)
    
    # 添加滑动平均
    window_size = 100
    if len(ep_rewards) > window_size:
        moving_avg = np.convolve(ep_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(ep_rewards)), moving_avg, label=f"{window_size}-Episode Moving Avg", color='red')
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Reward Curve")
    plt.legend()
    
    # 子图2: 完成时间对比
    plt.subplot(1, 2, 2)
    x_axis = np.arange(len(ppo_times)) * eval_interval
    plt.plot(x_axis, ppo_times, label="PPO Complete Time", marker='o')
    plt.plot(x_axis, mrass_times, label="MRASS Complete Time", marker='s')
    plt.xlabel("Episode")
    plt.ylabel("Complete Time")
    plt.title("PPO vs MRASS Complete Time")
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图片
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./model/ppo/ppo_training_curves_{current_date}.png")
    plt.show()

if __name__ == "__main__":
    main()
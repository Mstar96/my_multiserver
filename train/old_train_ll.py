# src/train_ll_dqn.py
import os
import numpy as np
from tqdm import tqdm,trange
import torch
import random
import sys
import matplotlib.pyplot as plt
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.old_single_server import SingleServerAllocEnv
from agent.old_dqn_agent import DQNAgent
from utils.marss import mrass_allocate 
from utils.load_dataset import load_dataset

def evaluate_policy(env, agent, tasks, alphas, C, episodes=1, eps=0.01):
    """评估策略性能"""
    results = []
    for _ in range(episodes):
        # 重置环境到指定任务
        env.reset(thread_lis=tasks, alphas=alphas)
        s = env.get_state()
        done = False
        while not done:
            a = agent.act(s, eps)
            s, r, done, info = env.step(a)
        allocation = info.get("allocation", None)
        complete_time = info.get("complete_time", None)
        results.append((allocation, complete_time))
    return allocation, complete_time

def final_evaluation(agent, env, verify_data, C):
    """最终评估函数"""
    dqn_times = []
    mrass_times = []
    
    for item in verify_data:
        tasks = item['thread_list']
        alphas = item['fi_func']
        # 评估DQN策略
        dqn_allocation, dqn_time = evaluate_policy(env, agent, tasks, alphas, C, episodes=1, eps=0.0)
        
        # 评估MRASS策略
        alloc_mrass, mrass_time = mrass_allocate(tasks, C, alphas)
        
        dqn_times.append(dqn_time)
        mrass_times.append(mrass_time)
    
    # 计算统计信息
    dqn_avg = np.mean(dqn_times)
    mrass_avg = np.mean(mrass_times)
    improvement = (mrass_avg - dqn_avg) / mrass_avg * 100
    
    print(f"最终评估结果:")
    print(f"DQN平均完成时间: {dqn_avg:.4f}")
    print(f"MRASS平均完成时间: {mrass_avg:.4f}")
    print(f"性能提升: {improvement:.2f}%")
    
def plot_training_curves(ep_rewards, dqn_times, mrass_times):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # 奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(ep_rewards)
    plt.title("训练奖励曲线")
    plt.xlabel("Episode")
    plt.ylabel("奖励")
    
    # 完成时间曲线
    plt.subplot(1, 2, 2)
    epochs = range(0, len(dqn_times) * 50, 50)
    plt.plot(epochs, dqn_times, label="DQN")
    plt.plot(epochs, mrass_times, label="MRASS")
    plt.title("验证集完成时间")
    plt.xlabel("Epoch")
    plt.ylabel("平均完成时间")
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图像
    current_date = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    plt.savefig(f"training_curves_{current_date}.png")
    plt.show()
    
def main():
    # 加载训练和验证数据
    train_data = load_dataset("data/data")
    verify_data = load_dataset("data/verify")
    
    # 使用第一个数据项初始化环境
    first_item = train_data[0]
    tasks = first_item['thread_list']
    alphas = first_item['fi_func']
    C = 100
    
    # 初始化环境
    env = SingleServerAllocEnv(tasks, alphas, C)
    state_dim = env.get_state().shape[0]
    action_dim = len(tasks)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99,
                     buffer_size=50000, batch_size=512, device=device)
    
    # 训练参数
    total_epochs = 100
    eps_start, eps_end = 1.0, 0.05
    eps_decay_steps = 5000
    update_every = 4
    target_update_every_eps = 20

    # 存放训练曲线
    ep_rewards = []
    dqn_times = []
    mrass_times = []
    best_dqn_time = float('inf')  # 初始化最佳时间
    
    global_step = 0
    
    # 训练循环
    for epoch in trange(total_epochs,desc="Epoch progress"):
        epoch_reward = 0.0
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (global_step / eps_decay_steps))
        random.shuffle(train_data)
        
        # 遍历训练集
        for item in tqdm(train_data,desc=f"Training in Epoch{epoch+1}",leave=False):
            tasks = item['thread_list']
            alphas = item['fi_func']
            
            # 重置环境到当前任务
            env.reset(thread_lis=tasks, alphas=alphas)
            state = env.get_state()
            done = False
            episode_reward = 0.0
            
            while not done:
                action = agent.act(state, eps)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                global_step += 1

                # 定期更新网络
                if global_step % update_every == 0:
                    _ = agent.update()
            
            epoch_reward += episode_reward
            ep_rewards.append(episode_reward)

        # 计算当前epoch的平均奖励
        avg_epoch_reward = epoch_reward / len(train_data)
        
        # 更新目标网络
        if epoch % target_update_every_eps == 0:
            agent.update_target()
                
        # 定期评估（每10个epoch）
        if (epoch + 1) % 10 == 0:
            dqn_avg_time = 0
            mrass_avg_time = 0
            
            for item in verify_data:
                tasks = item['thread_list']
                alphas = item['fi_func']
                
                dqn_allocation, dqn_time = evaluate_policy(env, agent, tasks, alphas, C, episodes=1, eps=0.0)
                alloc_mrass, mrass_time = mrass_allocate(tasks, C, alphas)
                
                dqn_avg_time += dqn_time
                mrass_avg_time += mrass_time
            
            dqn_avg_time /= len(verify_data)
            mrass_avg_time /= len(verify_data)

            dqn_times.append(dqn_avg_time)
            mrass_times.append(mrass_avg_time)

            print(f"Epoch {epoch+1}: DQN平均时间={dqn_avg_time:.4f}, "
                  f"MRASS平均时间={mrass_avg_time:.4f}, "
                  f"平均奖励={avg_epoch_reward:.4f}")
            
            # 如果性能更好，保存模型
            if dqn_avg_time < best_dqn_time:
                best_dqn_time = dqn_avg_time
                save_dir = "./model"
                os.makedirs(save_dir, exist_ok=True)
                current_date = datetime.now().strftime("%Y%m%d_%H_%M_%S")
                model_path = f"{save_dir}/dqn_ll_epoch_{epoch+1}_{current_date}.pt"
                torch.save(agent.q.state_dict(), model_path)
                print(f"保存新最佳模型: {model_path}")

    # 最终评估（在验证集上）
    final_evaluation(agent, env, verify_data, C)
    
    # 绘制训练曲线
    plot_training_curves(ep_rewards, dqn_times, mrass_times)

if __name__ == "__main__":
    main()
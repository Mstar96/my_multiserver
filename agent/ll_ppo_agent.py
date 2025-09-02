import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # 共享的特征提取层
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络 - 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络 - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        shared_features = self.shared_net(x)
        action_probs = self.actor(shared_features)
        state_values = self.critic(shared_features)
        return action_probs, state_values

class PPOBuffer:
    """PPO经验回放缓冲区"""
    def __init__(self, state_dim, action_dim, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.pointer = 0
        self.is_full = False
        
        # 初始化缓冲区
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        
    def store(self, state, action, reward, done, log_prob, value):
        """存储经验"""
        idx = self.pointer
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        
        self.pointer += 1
        if self.pointer >= self.buffer_size:
            self.pointer = 0
            self.is_full = True
            
    def get_batches(self):
        """获取批次数据"""
        buffer_size = self.buffer_size if self.is_full else self.pointer
        indices = np.random.choice(buffer_size, self.batch_size, replace=False)
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.LongTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.dones[indices]),
            torch.FloatTensor(self.log_probs[indices]),
            torch.FloatTensor(self.values[indices])
        )
        
    def clear(self):
        """清空缓冲区"""
        self.pointer = 0
        self.is_full = False

class PPOAgent:
    """PPO智能体"""
    def __init__(self, env, state_dim, action_dim, 
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, ppo_epochs=4, batch_size=64,
                 buffer_size=2048, hidden_dim=128):
        
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络和优化器
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验缓冲区
        self.buffer = PPOBuffer(state_dim, action_dim, buffer_size, batch_size)
        
    def get_action(self, state, deterministic=False):
        """根据状态选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.policy(state_tensor)
            
        if deterministic:
            action = torch.argmax(action_probs).item()
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            
        log_prob = torch.log(action_probs[0, action] + 1e-9)
        
        return action, log_prob.cpu().numpy(), value.squeeze().cpu().numpy()
    
    def compute_advantages(self, rewards, values, dones, next_value):
        """计算GAE优势函数"""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_advantage = 0
        next_value = next_value
        
        # 反向计算优势
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_values = values[t+1]
                
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantages[t]
            
        returns = advantages + values
        return advantages, returns
    
    def update(self, next_value):
        """更新策略"""
        if not self.buffer.is_full and self.buffer.pointer < self.batch_size:
            return  # 没有足够的样本进行更新
            
        # 获取缓冲区中的所有经验
        buffer_size = self.buffer_size if self.buffer.is_full else self.buffer.pointer
        
        states = torch.FloatTensor(self.buffer.states[:buffer_size]).to(self.device)
        actions = torch.LongTensor(self.buffer.actions[:buffer_size]).to(self.device)
        rewards = torch.FloatTensor(self.buffer.rewards[:buffer_size]).to(self.device)
        dones = torch.FloatTensor(self.buffer.dones[:buffer_size]).to(self.device)
        log_probs_old = torch.FloatTensor(self.buffer.log_probs[:buffer_size]).to(self.device)
        values_old = torch.FloatTensor(self.buffer.values[:buffer_size]).to(self.device)
        
        # 计算优势和回报
        advantages, returns = self.compute_advantages(
            rewards.cpu().numpy(), 
            values_old.cpu().numpy(), 
            dones.cpu().numpy(), 
            next_value
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮PPO更新
        for _ in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(buffer_size)
            
            for start in range(0, buffer_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 获取小批量数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                
                # 计算新策略的输出
                action_probs, values = self.policy(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                log_probs_new = dist.log_prob(batch_actions.squeeze())
                entropy = dist.entropy().mean()
                
                # 计算比率
                ratios = torch.exp(log_probs_new - batch_log_probs_old)
                
                # 计算裁剪的替代目标
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # 策略损失
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = 0.5 * (values.squeeze() - batch_returns).pow(2).mean()
                
                # 总损失
                loss = policy_loss + value_loss - 0.01 * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
        # 清空缓冲区
        self.buffer.clear()
        
    def train(self, num_episodes, max_steps=None):
        """训练循环"""
        if max_steps is None:
            max_steps = self.env.C  # 默认最大步数为资源数C
            
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            while not done and step < max_steps:
                # 选择动作
                action, log_prob, value = self.get_action(state)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.buffer.store(state, action, reward, done, log_prob, value)
                
                # 更新状态
                state = next_state
                episode_reward += reward
                step += 1
                
            # 计算下一个状态的价值（用于优势计算）
            if done:
                next_value = 0.0
            else:
                _, _, next_value = self.get_action(state)
                
            # 更新策略
            self.update(next_value)
            
            # 记录奖励
            episode_rewards.append(episode_reward)
            
            # 打印训练进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode+1}/{num_episodes}, Average Reward: {avg_reward:.4f}")
                
        return episode_rewards
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
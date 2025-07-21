import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """初始化Q网络"""
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """前向传播计算Q值"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        """初始化经验回放缓冲区"""
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """从缓冲区随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, buffer_capacity=10000, 
                 batch_size=64, target_update_freq=100):
        """初始化DQN智能体"""
        # Q网络和目标网络
        self.policy_net = DQN(state_dim, action_dim, hidden_dim)
        self.target_net = DQN(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络设为评估模式
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 超参数
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon_start  # 探索率
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq  # 目标网络更新频率
        self.action_dim = action_dim
        
        # 计数器
        self.step_count = 0
    
    def select_action(self, state, evaluate=False):
        """根据ε-贪婪策略选择动作"""
        if not evaluate and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.action_dim)
        else:
            # 利用：选择Q值最大的动作
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax(1).item()
    
    def update(self):
        """执行一次网络更新"""
        if len(self.replay_buffer) < self.batch_size:
            return  # 经验不足，暂不更新
        
        # 从经验回放缓冲区采样
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)
        
        # 计算当前Q值
        q_values = self.policy_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_net(next_state)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + (1 - done) * self.gamma * next_q_value
        
        # 计算损失
        loss = self.criterion(q_value, target_q_value)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

# 使用示例
def train_dqn(env, state_dim, action_dim, episodes=1000):
    """训练DQN智能体"""
    agent = DQNAgent(state_dim, action_dim)
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # 更新网络
            loss = agent.update()
            
            # 转移到下一状态
            state = next_state
            episode_reward += reward
        
        rewards_history.append(episode_reward)
        
        # 打印训练信息
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, rewards_history

# 注意：使用时需要导入相应的环境库，如OpenAI Gym
# 例如：import gym
# env = gym.make('CartPole-v1')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# agent, rewards = train_dqn(env, state_dim, action_dim)    
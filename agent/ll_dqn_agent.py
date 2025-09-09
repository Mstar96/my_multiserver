import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
    
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size1, hidden_size2, target_update, learning_rate, gamma, epsilon_decay, weight_decay, device):
        self.q_network = QNetwork(state_size, hidden_size1, hidden_size2, action_size)
        self.target_q_network = QNetwork(state_size, hidden_size1, hidden_size2, action_size) 
        self.q_network.to(device)
        self.target_q_network.to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.device = device
        self.epsilon_decay = epsilon_decay
        self.epsilon = 1.0
        self.action_size = action_size
        self.count = 0
        self.target_update = target_update
        total_params = sum(p.numel() for p in self.q_network.parameters())
        print(f"Total parameters in the network: {total_params}")
 
    def select_action(self, state, available_actions):
        if np.random.rand() < self.epsilon:
            # 随机选择未被选择的动作
            avail_actions = torch.arange(0, self.action_size, dtype=torch.int64)[available_actions]
            return np.random.choice(avail_actions.cpu())
        else:
            with torch.no_grad():
                q_values = self.q_network(state.view(1, -1))
                q_values[~available_actions.view(1, -1)] = -float('inf')
                return torch.argmax(q_values).item()

    
    def train(self, transition_dict):
        states = transition_dict['states'].float().view(1, -1).to(self.device)
        actions = torch.tensor([transition_dict['actions']], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor([transition_dict['rewards']], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = transition_dict['next_states'].float().view(1, -1).to(self.device)
        dones = torch.tensor([transition_dict['dones']], dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_network(states).gather(1, actions)
        max_next_q_values = self.target_q_network(next_states).max(1)[0].view(-1, 1)
        
        with torch.no_grad():
            # 计算目标 Q 值
            target_q_values = q_values.clone()
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        
        if self.count % self.target_update == 0:
            self.update_target_network()
        self.count += 1

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())#
import torch
## from machine import Machine
from utils.machine import Machine

class ResourceAllocationEnvironment:
    def __init__(self, machines, max_resource, stepper, device):
        func = Machine(9000)
        self.stepper = stepper#每一步分配的资源数量
        self.machines = machines   # 线程类型序列 
        self.num_items = len(machines)  # 线程数量
        self.machine_list = func.get_funclist(machines)# 线程的性能函数
        self.max_resource = max_resource# 总资源数
        
        self.items = torch.stack([torch.tensor(machines, device=device), torch.zeros(self.num_items, device=device)], dim=1)
        self.current_state = torch.cat([self.items, torch.tensor([[0, max_resource]], device=device)], dim=0)
        self.device = device
    
    def render(self):
        print(f"机器类型: {self.machines}")
        print(f"总资源: {self.max_resource}")
        print(f"状态:\n{self.items.cpu().numpy()}")
        print(f"当前状态:\n{self.current_state.cpu().numpy()}")

    def reset(self):
        """
          重置环境
        """
        self.items = torch.stack([torch.tensor(self.machines, device=self.device), torch.ones(self.num_items, device=self.device)], dim=1)
        self.current_state = torch.cat([self.items, torch.tensor([[0, self.max_resource]], device=self.device)], dim=0)
        return self.current_state
    
    def computePerformance(self, machine):
        """
        计算线程获取一个基数的资源后所获得的性能提升
        """
        performance = self.machine_list[int(machine[0])-1](machine[1] + self.stepper) \
                - self.machine_list[int(machine[0])-1](machine[1])
        if performance <= 0:
            return -1
        else:
            return performance
        
    def step(self, action):
        """
        进行一步动作
        """
        if action < 0 or action >= self.num_items:
            raise ValueError("Invalid action")

        reward = self.computePerformance(self.current_state[action])  # 返回性能提升值作为奖励
        self.current_state[action][1] = self.current_state[action][1] + self.stepper # 增加该线程已使用资源
        self.current_state[self.num_items][1] = self.current_state[self.num_items][1] - self.stepper  # 减少剩余资源数
            
#         self.current_resource += 0.1
#         reward = self.computePerformance(self.current_state[action])  # 返回性能提升值作为奖励
#         self.current_state[action][1] = self.current_state[action][1] + 0.1 # 增加该线程已使用资源
#         self.current_state[self.num_items][1] = self.current_state[self.num_items][1] - 0.1  # 减少剩余资源数
        done = self.current_state[self.num_items][1] <= 0
        
        return self.current_state, reward, done
    
    
    def step2(self, action):
        if action < 0 or action >= self.num_items:
            raise ValueError("Invalid action")

        self.current_state[action][1] = self.current_state[action][1] + self.stepper # 增加该线程已使用资源
        self.current_state[self.num_items][1] = self.current_state[self.num_items][1] - self.stepper  # 减少剩余资源数
            
        done = self.current_state[self.num_items][1] <= 0
        
        return self.current_state, done
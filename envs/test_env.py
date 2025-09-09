import torch
import random

class Env:
    def __init__(self,server_num,thread_num,capacity,device = "cuda"):
        self.server_num = server_num
        self.thread_num = thread_num
        self.servers =  [{"id":i,"capacity": capacity,"used":0} for i in range(server_num)] 
        self.threads =  [{"id":i,"demand":random.randint(1,5)} for i in range(thread_num)] 
        self.capacity = capacity
        self.device = device
        self.U = set(range(thread_num))
        self.allocation = {}
         
    def reset(self,capacity):
        self.capacity = capacity
        for server in self.servers:
            server["used"] = 0
            server["capacity"] = self.capacity
        for thread in self.threads:
            thread["demand"] = random.randint(1,5)

        self.U = set(range(self.thread_num))
        self.allocation = {}

    
    def step(self,thread_id=None,server_id=None):
         # 如果所有线程都已分配，则结束
        if not self.U:
            return True, 0, True, "所有线程已分配完毕"
        
        # 选择线程
        if thread_id is None:
            thread_id = random.choice(list(self.U))
        elif thread_id not in self.U:
            return False, -1, False, f"线程 {thread_id} 已被分配或不存在"
        
        # 选择服务器
        if server_id is None:
            server_id = random.randint(0, self.server_num - 1)
        elif server_id >= self.server_num:
            return False, -1, False, f"服务器 {server_id} 不存在"

        thread_demand = self.threads[thread_id]["demand"]
        server_remaning = self.servers[server_id]["capacity"] - self.servers[server_id]["used"]
 # 检查是否满足容量约束
        if thread_demand <= server_remaning:
            # 分配成功
            self.servers[server_id]["used"] += thread_demand
            self.allocation[thread_id] = server_id
            self.U.remove(thread_id)
            
            # 检查是否所有线程都已分配
            done = len(self.U) == 0
            return True, 1, done, f"线程 {thread_id} 成功分配到服务器 {server_id}"
        else:
            # 分配失败，容量不足
            return False, -1, False, f"服务器 {server_id} 容量不足，无法分配线程 {thread_id}"
            
    def get_state(self):
        """返回状态"""
        server_usage = [s["used"] / s["capacity"] for s in self.servers]
        return{
            "server_usage": server_usage,
            "remaning_threads": len(self.U),
            "allocation":self.allocation.copy()
        }
    def render(self):
        """打印当前环境状态"""
        print("=== 环境状态 ===")
        print(f"服务器数量: {self.servers}")
        print(f"线程数量: {self.threads}")
        print(f"服务器容量: {self.capacity}")
        
        print("\n服务器状态:")
        for i, server in enumerate(self.servers):
            print(f"  服务器 {i}: 已使用 {server['used']}/{server['capacity']}")
        
        print("\n线程状态:")
        for i, thread in enumerate(self.threads):
            status = "已分配" if i not in self.U else "未分配"
            server = self.allocation.get(i, "无")
            print(f"  线程 {i}: 需求={thread['demand']}, 状态={status}, 服务器={server}")
        
        print(f"\n未分配线程数: {len(self.U)}")
        print("================")

if __name__ == "__main__":
    envs = Env(2,4,30,"cuda")
    envs.render()
    for step in range(5):
        print(f"步骤：{step + 1}:")
        success,reward,done,info = envs.step()
        print(f"结果:{info},奖励:{reward}")
        envs.render()

        if done:
            print("所有线程已经分配完成")
            envs.render() 
            break

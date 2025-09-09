import torch
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import matplotlib.pyplot as plt
from pylab import mpl
from envs.single_server import ResourceAllocationEnvironment
from agent.ll_dqn_agent import DQNAgent
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import datetime
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备：{device}")
m_size = 5
machines = np.random.randint(1, 5, size=m_size)
max_resource = 30000
stepper = 100
env = ResourceAllocationEnvironment(machines, max_resource, stepper, device)
env.render()
agent = DQNAgent(env.current_state.numel(),len(machines),1024,1024,10,0.001,1,0.996,1e-6,"cuda")

def dqn(env,agent):
    total_reward = 0
    total_state = []
    
    state = env.reset()
    while True:
        action = agent.se 



if __name__ == "__main__":
    dqn(env,agent)
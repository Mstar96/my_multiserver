import numpy as np
import torch
from tqdm import trange
import sys, os
import matplotlib.pyplot as plt   # ✅ 新增：画图用
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.multi_server import MultiServerEnv
from envs.single_server import SingleServerAllocEnv
from agent.ll_dqn_agent import DQNAgent       # 低层 DQN
from agent.hl_dqn_agent import HLDQNAgent     # 高层 DQN
# from agent.hl_ppo_agent import HLPPOAgent   # 高层 PPO 可替换

#训练脚本


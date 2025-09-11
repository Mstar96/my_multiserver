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

thread_list = [449, 486, 410]
fi_alphas = [0.74, 0.79, 0.75]
C = 100
env = SingleServerAllocEnv(thread_list,fi_alphas,C)
state_dim = env.get_state().shape[0]
action_dim = len(thread_list)
print(f"state_dim:{state_dim}\naction_dim:{action_dim}")
train_data = load_dataset("data/data")
for item in train_data:
    print(f"idx:{item['id']},tasks:{item['thread_list']},func:{item['fi_func']}")
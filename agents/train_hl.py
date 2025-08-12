## 高层训练脚本（PPO），在训练时调用 MRASS 或已训练的 LL。
# agents/train_hl.py
import os
import gym
import numpy as np
from stable_baselines3 import PPO
from envs.multi_server_env import MultiServerEnv
from utils.ll_wrapper import LLWrapper

def train_hl(total_timesteps=200_000, use_ll=False, ll_model_path=None):
    env = MultiServerEnv(num_servers=6, num_threads=18, max_resource=60, power_P=1.0)
    if use_ll and ll_model_path is not None:
        ll = LLWrapper(ll_model_path, max_resource=60)
    else:
        ll = None

    # wrapper env that calls LL when step called
    class HLGymWrapper(gym.Env):
        def __init__(self, base_env, ll):
            super().__init__()
            self.base = base_env
            self.ll = ll
            self.observation_space = base_env.observation_space
            self.action_space = base_env.action_space

        def reset(self):
            return self.base.reset()

        def step(self, action):
            return self.base.step(action, ll_agent=self.ll)

        def render(self, mode='human'):
            return self.base.render(mode)

    hl_env = HLGymWrapper(env, ll)
    model = PPO("MultiInputPolicy", hl_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    os.makedirs("models", exist_ok=True)
    model.save("models/hl_ppo")
    return model

if __name__ == "__main__":
    train_hl()

##低层训练脚本（SAC），训练一个可复用的 LL policy。
# agents/train_ll.py
import os
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from envs.server_alloc_env import ServerAllocationEnv
import torch

def train_ll(total_timesteps=200_000, save_path="models/ll_sac"):
    # vectorized env of identical distribution (k random)
    def make_env():
        return ServerAllocationEnv(max_resource=60)
    venv = make_vec_env(make_env, n_envs=4)
    model = SAC("MlpPolicy", venv, verbose=1, tensorboard_log="./tb_logs/ll", device="cpu")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(save_path, exist_ok=True)
    model.save(os.path.join(save_path, "sac_ll"))
    return model

if __name__ == "__main__":
    train_ll()

# src/train_ll_dqn.py
import os
import numpy as np
from tqdm import trange
import torch

from envs.single_server import SingleServerAllocEnv
from agent.ll_dqn_agent import DQNAgent
from utils.marss import mrass_allocate  # 你的 mrass implementation

def evaluate_policy(env, agent, episodes=10, eps=0.01):
    """Evaluate greedy policy (eps small). Return average final M and allocation."""
    Ms = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s, eps)
            s, r, done, info = env.step(a)
        Ms.append(info["M"])
    return np.mean(Ms), Ms

def main():
    # example problem (你可替换为随机批次或dataset)
    thread_list = [449, 486, 410]
    fi_alphas = [0.74, 0.79, 0.75]
    C = 30  # 根据你示例分配结果为 [11,10,9] 时 C=30

    env = SingleServerAllocEnv(thread_list, fi_alphas, C, seed=0)
    state_dim = env._get_state().shape[0]
    action_dim = len(thread_list)

    device = "cpu"
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99,
                     buffer_size=50000, batch_size=64, device=device)

    # hyperparams
    eps_start, eps_end = 1.0, 0.05
    eps_decay_steps = 20000
    total_episodes = 2000
    update_every = 4
    target_update_every_eps = 20

    global_step = 0
    for ep in trange(total_episodes, desc="Train DQN-LL"):
        s = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # linear eps
            eps = max(eps_end, eps_start - (eps_start - eps_end) * (global_step / eps_decay_steps))
            # choose action
            a = agent.act(s, eps)
            ns, r, done, info = env.step(a)
            agent.remember(s, a, r, ns, done)
            s = ns
            ep_reward += r
            global_step += 1

            # update
            if global_step % update_every == 0:
                _ = agent.update()

        # update target periodically
        if ep % target_update_every_eps == 0:
            agent.update_target()

        # periodic eval & compare to MRASS
        if (ep+1) % 100 == 0:
            # evaluate DQN policy
            meanM, Ms = evaluate_policy(env, agent, episodes=30, eps=0.0)
            # MRASS baseline (call once)
            alloc_mrass, M_mrass = mrass_allocate(thread_list, C, fi_alphas)
            print(f"EP {ep+1}: DQN mean M={meanM:.4f}, MRASS M={M_mrass:.4f}, MRASS alloc={alloc_mrass}, sample Ms={Ms[:5]}")
            # optional: save model
            save_dir = "checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(agent.q.state_dict(), f"{save_dir}/dqn_ll_ep{ep+1}.pt")

    # final evaluation
    meanM, Ms = evaluate_policy(env, agent, episodes=200, eps=0.0)
    alloc_mrass, M_mrass = mrass_allocate(thread_list, C, fi_alphas)
    print("Final: DQN mean M=", meanM, " MRASS M=", M_mrass, "MRASS alloc=", alloc_mrass)

if __name__ == "__main__":
    main()

## 主环境（Gym 风格），高层（HL）与低层（LL）接口、内置 MRASS oracle（单服资源分配）
#主环境。High-level API：HL 做 assignment（action = server id），环境调用 LL （若提供）或 MRASS 来分配资源并返回即时 reward（-ΔE）。为简单起见，线程完成不模拟实际时间流（we compute M_j after allocations).
# envs/multi_server_env.py
import gym
from gym import spaces
import numpy as np
from utils.mrass import mrass_allocate, fi_power

class MultiServerEnv(gym.Env):
    """
    HL-facing environment.
    Observation: dict with
      - current_thread: [li_norm, alpha]
      - servers_summary: for each server [allocated_resource_sum, M_j]
    Action: discrete server index to assign current thread
    Reward: -delta_energy = -P * (new_M_j - old_M_j) for affected server
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_servers=6, num_threads=18, max_resource=60, power_P=1.0, seed=None):
        super(MultiServerEnv, self).__init__()
        self.num_servers = num_servers
        self.num_threads = num_threads
        self.max_resource = max_resource
        self.power_P = power_P
        self.rng = np.random.RandomState(seed)

        # thread params: each thread has li (int) and alpha
        # observation spaces
        self.observation_space = spaces.Dict({
            "current_thread": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "servers": spaces.Box(low=0.0, high=max_resource, shape=(num_servers, 2), dtype=np.float32),
            "remaining": spaces.Box(low=0, high=num_threads, shape=(1,), dtype=np.int32)
        })
        self.action_space = spaces.Discrete(num_servers)

        # state
        self.reset()

    def _sample_threads(self):
        # li in [100,200], alpha in (0.2,1.0)
        lis = self.rng.randint(100, 201, size=self.num_threads).astype(np.float32)
        alphas = self.rng.uniform(0.2, 1.0, size=self.num_threads).astype(np.float32)
        return lis, alphas

    def reset(self):
        self.lis, self.alphas = self._sample_threads()
        # Threads assigned per server: lists
        self.server_threads = [[] for _ in range(self.num_servers)]
        # resource allocations per server (list of ints per thread)
        self.server_allocs = [[] for _ in range(self.num_servers)]
        # server M_j (float)
        self.server_M = [0.0 for _ in range(self.num_servers)]
        # index of next thread to assign
        self.cur_idx = 0
        # precompute fi functions for each thread
        self.fi_funcs = [ (lambda a: (lambda c, a=a: fi_power(c, a)))(alpha) for alpha in self.alphas ]
        return self._get_obs()

    def _get_obs(self):
        cur_thread = np.array([self.lis[self.cur_idx]/200.0, (self.alphas[self.cur_idx]-0.2)/0.8], dtype=np.float32)
        servers_arr = np.zeros((self.num_servers, 2), dtype=np.float32)
        for j in range(self.num_servers):
            # allocated sum normalized, and M_j normalized by an upper bound
            alloc_sum = sum(self.server_allocs[j]) if len(self.server_allocs[j])>0 else 0.0
            servers_arr[j,0] = alloc_sum / max(1.0, self.max_resource)
            # rough normalize M_j by 200/ (f(1) min ) -> for simplicity divide by 200
            servers_arr[j,1] = self.server_M[j] / 200.0
        remaining = np.array([self.num_threads - self.cur_idx], dtype=np.int32)
        return {"current_thread": cur_thread, "servers": servers_arr, "remaining": remaining}

    def step(self, action, ll_agent=None):
        """
        action: server id
        ll_agent: optional callable: allocate_func(server_threads, C, fi_funcs_subset) -> list of ints allocs, M_j
                  if None, we use MRASS oracle on that server
        returns obs, reward, done, info
        """
        server_id = int(action)
        # prepare server thread lists after adding this thread
        t_li = float(self.lis[self.cur_idx])
        t_alpha = float(self.alphas[self.cur_idx])
        # Append thread temporarily and compute new allocation & M
        new_thread_list = [*self.server_threads[server_id], t_li]
        # fi funcs for threads on that server (existing + new)
        old_fi = []
        for idx, li in enumerate(self.server_threads[server_id]):
            # find original global index? we stored only lis; we'll approximate by using alpha samples of current threads
            # Simpler: use same alpha distribution - map by index in global lists not tracked => in this simplified impl
            # we instead reconstruct fi func based on nearest alpha (practical: track global indices per server)
            pass
        # To keep mapping correct, we will store global idx for server threads rather than just lis
        # (Thus refactor: server_threads store indices)
        # For now do quick refactor below: (we'll maintain server_threads as lists of indices)

        # ---- Minimal refactor start ----
        # (We assume server_threads store indices; in reset we should initialize empty lists of indices.)
        # For safety, if server_threads contains floats (old code), convert:
        try:
            # if server_threads entries are indices ints, proceed
            _ = self.server_threads[server_id]
        except:
            raise RuntimeError("server_threads must store indices of threads")

        # compute fi_funcs list for that server
        indices = self.server_threads[server_id] + [self.cur_idx]
        fi_subset = [ self.fi_funcs[i] for i in indices ]
        li_subset = [ float(self.lis[i]) for i in indices ]

        # old M_j
        old_M = self.server_M[server_id]

        # call allocator
        if ll_agent is None:
            allocs, new_M = mrass_allocate(li_subset, self.max_resource, fi_subset)
        else:
            # ll_agent.allocate expects list of (li, alpha) or indices and returns allocation (same order)
            allocs, new_M = ll_agent.allocate(indices, li_subset, fi_subset, self.max_resource)

        # update state: add new thread index and update allocs for server
        self.server_threads[server_id].append(self.cur_idx)
        # replace server_allocs with allocs (we keep order matching indices)
        self.server_allocs[server_id] = allocs
        self.server_M[server_id] = new_M

        delta_M = new_M - old_M
        reward = - self.power_P * delta_M

        self.cur_idx += 1
        done = (self.cur_idx >= self.num_threads)
        obs = self._get_obs()
        info = {"server": server_id, "delta_M": delta_M, "new_M": new_M}
        return obs, reward, done, info

    def render(self, mode='human'):
        print("Server M:", self.server_M)
        print("Server alloc sums:", [sum(a) for a in self.server_allocs])

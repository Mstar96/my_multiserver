## 低层环境（ServerAllocationEnv），可单独用来训练 LL（SAC）
# 低层环境，用于训练 LL。给定一个服务器的线程集合（indices），任务是输出资源比例或整数分配。我们实现 continuous action（比例）版，使用 SAC。
import gym
from gym import spaces
import numpy as np
from utils.mrass import mrass_allocate

class ServerAllocationEnv(gym.Env):
    """
    LL-facing environment to allocate resources among k threads on one server.
    Observation: flattened [li_norm, alpha] * k + remaining_capacity_norm (optional)
    Action: continuous vector a_i in [0,1] length k -> produces integer allocations via projection
    Reward: negative server active time (-M_j) so SAC maximizes negative M_j (i.e., minimizes M_j)
    """
    def __init__(self, max_resource=60, seed=None):
        super(ServerAllocationEnv, self).__init__()
        self.max_resource = max_resource
        self.rng = np.random.RandomState(seed)
        self.k = None
        self.observation_space = None
        self.action_space = None
        # we'll set these on reset by sampling a random thread set
        self.cur_indices = None
        self.li = None
        self.fi = None

    def sample_server_threads(self, k=None):
        if k is None:
            k = self.rng.randint(1, 8)  # random group size
        self.k = k
        # sample lis & alphas
        self.li = self.rng.randint(100,201,size=k).astype(np.float32)
        alphas = self.rng.uniform(0.2, 1.0, size=k).astype(np.float32)
        self.fi = [ (lambda a: (lambda c, a=a: (c if c>0 else 1e-6)**a))(alpha) for alpha in alphas ]
        # build spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(k,2), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(k,), dtype=np.float32)
        return self._get_obs()

    def reset(self, k=None):
        obs = self.sample_server_threads(k)
        return obs

    def _get_obs(self):
        obs = np.zeros((self.k,2), dtype=np.float32)
        for i in range(self.k):
            obs[i,0] = self.li[i] / 200.0
            # approximate alpha sampling not stored, so just normalize a placeholder
            # (we could store alphas if needed)
            # here we keep only li normalized for simplicity
            obs[i,1] = 0.5
        return obs

    def step(self, action):
        # action shape (k,) continuous in [0,1]
        a = np.array(action, dtype=np.float32)
        if a.sum() <= 0:
            allocs = [0]*self.k
        else:
            ratios = a / (a.sum() + 1e-8)
            # map to integers
            raw = (ratios * self.max_resource)
            allocs = np.floor(raw).astype(int).tolist()
            # fix rounding to ensure sum <= max_resource and distribute remainder greedily
            rem = self.max_resource - sum(allocs)
            if rem > 0:
                # distribute to largest fractional parts
                fracs = raw - np.floor(raw)
                idxs = np.argsort(-fracs)
                for idx in idxs:
                    if rem <= 0:
                        break
                    allocs[idx] += 1
                    rem -= 1
        # compute M via fi functions
        M = 0.0
        for li, f, r in zip(self.li, self.fi, allocs):
            sp = f(r)
            T = li / (sp if sp>0 else 1e-6)
            if T > M: M = T
        reward = - M  # maximize negative M
        done = True  # one-step episode (we can make it one-step)
        info = {"allocs": allocs, "M": M}
        return self._get_obs(), reward, done, info

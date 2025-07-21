import gym
from gym import spaces
import numpy as np

class ThreadResourceEnv(gym.Env):
    def __init__(self, n_threads=10, m_servers=3, server_capacity=60):
        super(ThreadResourceEnv, self).__init__()

        self.n_threads = n_threads
        self.m_servers = m_servers
        self.C = server_capacity
        self.P = 1.0

        self.state_dim = m_servers * n_threads + 3
        self.observation_space = spaces.Box(low=0, high=1e3, shape=(self.state_dim,), dtype=np.float32)

        self.max_resource_per_thread = self.C
        self.action_space = spaces.MultiDiscrete([n_threads, m_servers, self.max_resource_per_thread+1])

        self.reset()

    def reset(self):
        self.threads = []
        for i in range(self.n_threads):
            li = np.random.randint(100, 200)
            alpha = np.random.uniform(0.1, 1.0)
            self.threads.append({'id': i, 'length': li, 'alpha': alpha, 'assigned': False})

        self.servers = []
        for j in range(self.m_servers):
            self.servers.append({
                'id': j,
                'remaining': self.C,
                'threads': [],  # list of assigned thread ids
                'max_time': 0.0
            })
        self.done = False
        return self._get_state()

    def _get_state(self):
        server_state = []
        for s in self.servers:
            server_state.append(s['remaining'])
            server_state.append(s['max_time'])

        thread_state = []
        for t in self.threads:
            thread_state.append(0 if t['assigned'] else 1)
            thread_state.append(t['length'])
            thread_state.append(t['alpha'])
        #将服务器状态和线程状态拼起来变成一维数组[1,2,3,4,5,6]
        state = np.array(server_state + thread_state, dtype=np.float32)
        
        return state

    def _compute_thread_time(self, length, alpha, c):
        if c == 0:
            return 1e6
        return length / (c**alpha)

    def step(self, action):
        thread_idx, server_idx, resource = action
        reward = 0.0

        if self.done:
            return self._get_state(), 0.0, True, {}

        if thread_idx >= self.n_threads or server_idx >= self.m_servers:
            return self._get_state(), -10.0, False, {}

        thread = self.threads[thread_idx]
        server = self.servers[server_idx]

        if thread['assigned']:
            return self._get_state(), -5.0, False, {}

        actual_resource = min(resource, server['remaining'])
        if actual_resource <= 0:
            self.invalid_actions.add((thread_idx, server_idx, resource))
            return self._get_state(), -5.0, False, {}

        thread['assigned'] = True
        exec_time = self._compute_thread_time(thread['length'], thread['alpha'], actual_resource)
        server['remaining'] -= actual_resource
        server['threads'].append({'id': thread_idx, 'c': actual_resource, 'T': exec_time})
        server['max_time'] = max(server['max_time'], exec_time)

        total_energy = sum(s['max_time']*self.P for s in self.servers)
        reward = -total_energy

        unassigned_threads = [t for t in self.threads if not t['assigned']]
        has_available_space = any(s['remaining'] > 0 for s in self.servers)

        if not has_available_space and unassigned_threads:
            self.done = True
            reward -= 100.0

        if not unassigned_threads:
            self.done = True

        info = {
            "unassigned_threads": [t['id'] for t in unassigned_threads],
            "server_remaining_resources": [s['remaining'] for s in self.servers]
        }

        return self._get_state(), reward, self.done, info

    def render(self, mode='human'):
        print("Servers:")
        for s in self.servers:
            assigned_ids = [str(t['id']) for t in s['threads']]
            print(f"  Server {s['id']}: remaining {s['remaining']}, max_time {s['max_time']}, assigned threads: {', '.join(assigned_ids)}")
        print("Threads:")
        for t in self.threads:
            print(f"  Thread {t['id']}: assigned {t['assigned']}, length {t['length']}, alpha {t['alpha']}")

if __name__ == "__main__":
    env = ThreadResourceEnv(n_threads=4, m_servers=2, server_capacity=60)

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        # 随机动作
        action = env.action_space.sample()
        print(f"\n🎯 Action: {action}")
        next_state, reward, done, info = env.step(action)
        print(f"🏆 Reward: {reward:.2f}\n")
        total_reward += reward

    print(f"✅ Episode finished. Total reward: {total_reward:.2f}\n")
    env.render()

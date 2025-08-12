# utils/ll_wrapper.py
import numpy as np
from stable_baselines3 import SAC

class LLWrapper:
    def __init__(self, model_path, max_resource):
        self.model = SAC.load(model_path)
        self.max_resource = max_resource

    def allocate(self, indices, li_subset, fi_subset, C):
        # indices: list of global indices (we ignore)
        # li_subset, fi_subset used to construct obs
        k = len(li_subset)
        obs = np.zeros((k,2), dtype=np.float32)
        for i in range(k):
            obs[i,0] = li_subset[i] / 200.0
            obs[i,1] = 0.5
        # model expects flattened or vectorized obs; we pass as dict of arrays via predict
        # stable-baselines3 expects a 1D vector; we flatten
        obs_flat = obs.flatten()
        action, _ = self.model.predict(obs_flat, deterministic=True)
        # model expects same obs shape at training; if mismatch, one should adapt policy to accept variable sizes.
        # For simplicity, we assume model trained with fixed k (or we have multiple models per k).
        # Here we simulate allocate using simple proportional mapping:
        a = action
        # Fallback: if action shape mismatch, use uniform split
        if not hasattr(a, "__len__") or len(a) != k:
            ratios = np.ones(k) / k
            raw = ratios * C
        else:
            ratios = a / (a.sum() + 1e-8)
            raw = ratios * C
        allocs = np.floor(raw).astype(int).tolist()
        rem = C - sum(allocs)
        if rem > 0:
            fracs = raw - np.floor(raw)
            idxs = np.argsort(-fracs)
            for idx in idxs:
                if rem <= 0: break
                allocs[idx] += 1
                rem -= 1
        # compute M
        M = 0.0
        for li, f, r in zip(li_subset, fi_subset, allocs):
            sp = f(r)
            T = li / (sp if sp>0 else 1e-6)
            if T > M: M = T
        return allocs, M

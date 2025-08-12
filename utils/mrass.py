#MRASS 的 Python 实现（用于 oracle / imitation targets）
# utils/mrass.py
import math
import numpy as np

def fi_power(c, alpha):
    """ 确定c大于0，如果c=0返回一个很小的 ε 速度以避免除零错误
    """
    return (c if c>0 else 1e-6) ** alpha

def find_resource_for_target_time(li, target_T, fi_func, C):
    """
    Find smallest integer r in [0,C] such that li / fi(r) <= target_T
    If none, return nearest r (C)
    """
    # binary search on r
    l, r = 0, C
    best = C
    while l <= r:
        mid = (l + r) // 2
        speed = fi_func(mid)
        if speed <= 0:
            time = float('inf')
        else:
            time = li / speed
        if time <= target_T:
            best = mid
            r = mid - 1
        else:
            l = mid + 1
    return best

def mrass_allocate(thread_list, C, fi_funcs):
    """
    thread_list: list of li (instruction lengths)
    fi_funcs: list of functions fi(c) -> speed (same length)
    Return allocation list of integers summing <= C and resulting M (max completion time)
    Implementation follows the MRASS high-level idea:
      - binary search on target T (TL..TR)
      - for a candidate T, compute required resources for each thread to reach <=T via find_resource_for_target_time
      - if sum <= C, try smaller T; else increase T
    """
    n = len(thread_list)
    # bounds for T: lower bound = max_i li / fi(C), upper bound = max_i li / fi(1) (if fi(1)>0)
    TL = 0.0
    TR = 0.0
    for li, f in zip(thread_list, fi_funcs):
        # handle fi(C) and fi(1)
        vC = f(C)
        v1 = f(1)
        TL = max(TL, li / (vC if vC>0 else 1e-6))
        TR = max(TR, li / (v1 if v1>0 else 1e-6))
    # search integer T seconds (we can use float and tolerance)
    eps = 1e-3
    best_alloc = None
    best_T = TR
    left, right = TL, TR
    iters = 0
    while right - left > eps and iters < 60:
        mid = (left + right) / 2.0
        reqs = []
        total = 0
        for li, f in zip(thread_list, fi_funcs):
            r = find_resource_for_target_time(li, mid, f, C)
            reqs.append(r)
            total += r
        if total <= C:
            best_alloc = reqs.copy()
            best_T = mid
            right = mid
        else:
            left = mid
        iters += 1
    if best_alloc is None:
        # fallback: greedy fill (give each at least 0, then distribute)
        best_alloc = [0]*n
        rem = C
        # give one by one to thread with highest marginal benefit (approx)
        while rem>0:
            # compute marginal reduction candidate: li/(f(r)) - li/(f(r+1))
            best_idx = None
            best_gain = -1
            for i in range(n):
                ri = best_alloc[i]
                cur = thread_list[i] / (fi_funcs[i](ri) if fi_funcs[i](ri)>0 else 1e-6)
                nxt = thread_list[i] / (fi_funcs[i](ri+1) if fi_funcs[i](ri+1)>0 else 1e-6)
                gain = cur - nxt
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            best_alloc[best_idx] += 1
            rem -= 1
    # ensure sum <= C
    s = sum(best_alloc)
    if s > C:
        # trim from smallest marginal benefit
        while s > C:
            worst_idx = None
            worst_loss = None
            for i in range(n):
                if best_alloc[i] == 0: continue
                ri = best_alloc[i]
                cur = thread_list[i] / (fi_funcs[i](ri) if fi_funcs[i](ri)>0 else 1e-6)
                prev = thread_list[i] / (fi_funcs[i](ri-1) if fi_funcs[i](ri-1)>0 else 1e-6)
                loss = prev - cur
                if worst_loss is None or loss < worst_loss:
                    worst_loss = loss
                    worst_idx = i
            best_alloc[worst_idx] -= 1
            s -= 1
    # compute M
    M = 0.0
    for li, f, r in zip(thread_list, fi_funcs, best_alloc):
        sp = f(r)
        T = li / (sp if sp>0 else 1e-6)
        if T > M: M = T
    return best_alloc, M

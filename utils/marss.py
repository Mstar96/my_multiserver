# src/utils/mrass.py
import numpy as np

def fi_power(c: int, alpha: float) -> float:
    """_summary_

    Args:
        c (int): 资源数
        alpha (float): 效用值

    Returns:
        float: 线程得到c资源时的效用函数值
    """
    if c <= 0: 
        return 1e-6
    return float(c) ** float(alpha)


def find_r_for_T(task: int,target_time: float, func: float, C: int) -> int:
    """_summary_

    Args:
        task (int): 线程的任务长度
        target_time (float): 目标时间，该线程需要接近的时间
        func (float): 效用函数值
        C (int): 总资源数

    Returns:
        resource(int): 该线程接近目标时间需要的资源数
    """
    l, r = 0, C
    best = C
    while l <= r:
        mid = (l + r) // 2
        v = fi_power(mid,func)
        time = task / (v if v > 0.0 else 1e-6)
        if time <= target_time:
            best = mid
            r = mid - 1
        else:
            l = mid + 1
    return int(best)

def mrass_allocate(thread_list, C: int, fi_funcs):
    """_summary_

    Args:
        thread_list (list(int)): 线程的任务长度
        C (int): 总资源数
        fi_funcs (float): 效用函数幂值

    Returns:
        allocation(list(int)): 资源分配方式
        time(float):最短完成任务时间
    """
    n = len(thread_list)
    TL, TR = 0.0, 0.0
    for li, f in zip(thread_list, fi_funcs):
        TL = max(TL, li / max(fi_power(C,f), 1e-6))
        TR = max(TR, li / max(fi_power(1,f), 1e-6))
    left, right = TL, TR
    best_alloc, best_T = None, TR
    for _ in range(60):
        mid = (left + right) / 2
        reqs = []
        total = 0
        for li, f in zip(thread_list, fi_funcs):
            r = find_r_for_T(li,mid,f,C)
            reqs.append(r)
            total += r
        if total <= C:
            best_alloc, best_T = reqs, mid
            right = mid
        else:
            left = mid
        if right - left < 1e-4:
            break
    if best_alloc is None:
        # fallback: 贪心（按边际收益发资源）
        alloc = [0]*n
        rem = C
        while rem > 0:
            best_i, best_gain = 0, -1.0
            for i in range(n):
                cur = thread_list[i] / max(fi_funcs[i](alloc[i]), 1e-6)
                nxt = thread_list[i] / max(fi_funcs[i](alloc[i]+1), 1e-6)
                gain = cur - nxt
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            alloc[best_i] += 1
            rem -= 1
        best_alloc = alloc
    # compute M
    M = 0.0
    for li, f, r in zip(thread_list, fi_funcs, best_alloc):
        M = max(M, li / max(fi_power(r,f), 1e-6))
    return best_alloc, float(M)

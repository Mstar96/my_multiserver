import random

def generate_test_data(num_threads: int, min_task: int, max_task: int, min_alpha: float, max_alpha: float):
    """
    随机生成测试数据
    Args:
        num_threads: 线程数量
        min_task: 任务量最小值
        max_task: 任务量最大值
        min_alpha: alpha值最小值（通常0~1之间，效用函数幂次）
        max_alpha: alpha值最大值
    Returns:
        thread_tasklist: 随机生成的任务量列表
        fi_funcs: 随机生成的alpha值列表（对应每个线程）
    """
    # 随机生成任务量（整数）
    thread_tasklist = [random.randint(min_task, max_task) for _ in range(num_threads)]
    
    # 随机生成alpha值（保留2位小数，符合效用函数幂次的常见范围）
    fi_funcs = [round(random.uniform(min_alpha, max_alpha), 2) for _ in range(num_threads)]
    
    return thread_tasklist, fi_funcs
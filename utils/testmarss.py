import marss as marss
import random
import matplotlib.pyplot as plt
import numpy as np
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

if __name__ == "__main__":
    num_threads = 3  # 线程数量
    min_task = 100   # 任务量下限
    max_task = 500   # 任务量上限
    min_alpha = 0.2  # alpha下限（建议0.1~1.0）
    max_alpha = 0.9  # alpha上限
    thread_tasklist , fi_funcs = generate_test_data(num_threads,min_task,max_task,min_alpha,max_alpha)
    C = 30
    allocate,time = marss.mrass_allocate(thread_tasklist,C,fi_funcs)
    print(f"thread_task_list = {thread_tasklist}\nfi_funcs = {fi_funcs}\nallocation = {allocate}\ntime = {time}")
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei"]
    x = np.arange(num_threads)  # 线程索引
    
    # 绘制任务量和分配情况对比图
    width = 0.35  # 柱状图宽度
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 任务量柱状图
    rects1 = ax1.bar(x - width/2, thread_tasklist, width, label='任务量')
    # 分配情况柱状图
    rects2 = ax1.bar(x + width/2, allocate, width, label='分配结果')
    
    ax1.set_xlabel('线程编号')
    ax1.set_ylabel('任务量/分配量')
    ax1.set_title('各线程任务量与分配情况对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'线程{i+1}' for i in range(num_threads)])
    ax1.legend()
    
    # 右侧坐标轴显示alpha值
    ax2 = ax1.twinx()
    ax2.plot(x, fi_funcs, 'ro-', label='效用函数值')
    ax2.set_ylabel('效用函数值')
    ax2.set_ylim(0, 1.0)  # alpha通常在0~1范围
    ax2.legend(loc='upper right')
    
    plt.text(0.85, 1.05, f'总完成时间：{time:.2f}', 
         transform=ax1.transAxes, 
         verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
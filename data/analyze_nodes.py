# analyze_nodes.py
import pandas as pd
import matplotlib.pyplot as plt

# --- 请将这里的文件路径指向您使用的原始日志文件 ---
LOGS_CSV_PATH = 'results/node_logs.csv'
# ----------------------------------------------------

def analyze_node_performance(csv_path):
    """
    读取日志文件，计算并打印每个节点的平均性能指标。
    """
    try:
        # 读取数据并为列命名
        df = pd.read_csv(csv_path, header=None, names=['node_id', 'cpu', 'memory', 'delay', 'load'])
    except FileNotFoundError:
        print(f"错误：文件未找到 '{csv_path}'。请确保路径正确。")
        return

    # 按节点ID分组，计算每个指标的平均值
    # 对于延迟和负载，低一些更好；对于CPU和内存，高一些更好
    node_avg_performance = df.groupby('node_id').agg({
        'cpu': 'mean',
        'memory': 'mean',
        'delay': 'mean',
        'load': 'mean'
    })

    print("--- 节点平均性能分析 ---")
    print(node_avg_performance)
    print("\n--- 结论 ---")
    
    # 检查是否存在明显的“超级节点”
    # 我们定义一个简单的检查：如果一个节点的延迟比平均延迟低20%以上，且CPU比平均CPU高20%以上，就可能是超级节点
    avg_delay = node_avg_performance['delay'].mean()
    avg_cpu = node_avg_performance['cpu'].mean()
    
    potential_super_nodes = node_avg_performance[
        (node_avg_performance['delay'] < avg_delay * 0.8) & 
        (node_avg_performance['cpu'] > avg_cpu * 1.2)
    ]
    
    if not potential_super_nodes.empty:
        print("警告：检测到潜在的'超级节点'，它们的平均性能显著优于其他节点：")
        print(potential_super_nodes)
    else:
        print("未检测到明显的'超级节点'。各个节点的平均性能较为均衡。")
        
    # 可视化对比
    node_avg_performance.plot(kind='bar', subplots=True, figsize=(12, 8), layout=(2, 2), legend=False)
    plt.suptitle('各节点平均性能指标对比', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/node_performance_analysis.png')
    print("\n图表已保存至 results/node_performance_analysis.png")
    plt.show()

if __name__ == '__main__':
    analyze_node_performance(LOGS_CSV_PATH)
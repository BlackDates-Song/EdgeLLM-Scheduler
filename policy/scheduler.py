from dataclasses import dataclass
import random

@dataclass
class Weights:
    alpha: float = 1.0  # 网络延迟权重
    beta: float = 1.0   # 计算时间权重
    margin_ms: float = 0.0  # 固定延迟裕量

def estimate_eta(pred, req_cost=1.0, w: Weights = Weights()):
    """根据预测值估算端到端延迟 (eta)"""
    cpu, mem, delay, load = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
    eff = max(1e-3, cpu * max(0.0, 1.0 - load))
    comp = req_cost / eff
    eta = w.alpha * delay + w.beta * comp
    eta += w.margin_ms
    return eta

def pick_node(scores, top_k=1, epsilon=0.0, threshold_ms=5.0):
    """
    根据最终得分，使用带安全阈值的Top-K epsilon-greedy策略选择节点。
    
    scores: 一个包含 {node_id: final_score} 的字典。
    top_k: 考虑分数最低的K个节点。
    epsilon: 在安全阈值内进行探索的概率。
    threshold_ms: 探索的安全阈值（毫秒）。

    返回选择的节点ID。
    """
    if not scores:
        raise ValueError("Scores dictionary cannot be empty.")

    # 按分数排序（越低越好）并获取Top-K
    sorted_nodes = sorted(scores.items(), key=lambda item: item[1])
    top_k_nodes = sorted_nodes[:top_k]
    
    best_node_tuple = top_k_nodes[0]
    
    # 默认选择最优节点
    chosen_node_id = best_node_tuple[0]

    # 安全检查：只有当候选节点足够多，且性能差距在阈值内时，才考虑探索
    if len(top_k_nodes) > 1 and random.random() < epsilon:
        # 获取所有在安全阈值内的候选节点（不包括最优的那个）
        safe_to_explore = [
            node_tuple for node_tuple in top_k_nodes[1:]
            if node_tuple[1] <= best_node_tuple[1] + threshold_ms
        ]
        
        if safe_to_explore:
            # 如果有安全的探索对象，则从中随机选择一个
            chosen_node_id = random.choice(safe_to_explore)[0]
            
    return chosen_node_id
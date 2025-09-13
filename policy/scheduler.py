import math
from dataclasses import dataclass
import random

@dataclass
class Weights:
    alpha: float = 1.0  # 网络延迟权重
    beta: float = 1.0   # 计算时间权重
    delay_rmse: float = 9.0 # 验证集Delay RMSE
    safety_k: float = 0.0  # 安全系数
    margin_ms: float = 0.0  # 固定延迟裕量

def estimate_eta(pred, req_cost=1.0, w: Weights = Weights()):
    """
    pred: [CPU, MEM, DELAY, LOAD] （**反归一化**后的真实量纲）
    req_cost: 请求计算体量（可理解为归一化 token 量/模型复杂度）
    """
    cpu, mem, delay, load = map(float, pred)
    eff = max(1e-3, cpu * max(0.0, 1.0 - load))  # 有效算力
    comp = req_cost / eff
    eta = w.alpha * delay + w.beta * comp
    # 安全裕量（防大偏差）
    eta += (w.margin_ms if w.margin_ms > 0 else w.safety_k * w.delay_rmse)
    return eta

def pick_node(predictions, req_cost=1.0, w: Weights = Weights(), top_k=1, epsilon=0.0):
    """
    Selects the best node using a Top-K epsilon-greedy strategy.
    predictions: dict[node_id] = [CPU, MEM, DELAY, LOAD]
    top_k: Consider the K best nodes.
    epsilon: Probability of choosing a random node from the K-1 non-best nodes.
    返回 (best_node, scores_dict)
    """
    scores = {
        nid: estimate_eta(pred, req_cost=req_cost, w=w) for nid, pred in predictions.items()
    }
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1])
    top_k_nodes = sorted_nodes[:top_k]

    if random.random() < epsilon and len(top_k_nodes) > 1:
        chosen = random.choice(top_k_nodes[1:])
    else:
        chosen = top_k_nodes[0]
    return chosen[0], scores
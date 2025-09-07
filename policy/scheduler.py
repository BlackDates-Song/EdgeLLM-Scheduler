import math
from dataclasses import dataclass

@dataclass
class Weights:
    alpha: float = 1.0  # 网络延迟权重
    beta: float = 1.0   # 计算时间权重
    delay_rmse: float = 9.0 # 验证集Delay RMSE
    safety_k: float = 0.3  # 安全系数

def estimate_eta(pred, req_cost=1.0, w: Weights = Weights()):
    """
    pred: [CPU, MEM, DELAY, LOAD] （**反归一化**后的真实量纲）
    req_cost: 请求计算体量（可理解为归一化 token 量/模型复杂度）
    """
    cpu, mem, delay, load = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
    eff = max(1e-3, cpu * max(0.0, 1.0 - load))  # 有效算力
    comp = req_cost / eff
    eta = w.alpha * delay + w.beta * comp
    # 安全裕量（防大偏差）
    eta += w.safety_k * w.delay_rmse
    return eta

def pick_node(predictions, req_cost=1.0, w: Weights = Weights()):
    """
    predictions: dict[node_id] = [CPU, MEM, DELAY, LOAD]
    返回 (best_node, scores_dict)
    """
    scores = {}
    for nid, pred in predictions.items():
        scores[nid] = estimate_eta(pred, req_cost=req_cost, w=w)
    best = min(scores, key=scores.get)
    return best, scores
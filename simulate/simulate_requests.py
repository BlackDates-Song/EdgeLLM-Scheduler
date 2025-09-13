# simulate/simulate_requests.py
import os
import csv
import math
import argparse
import random
import numpy as np
import torch
from torch import nn
from collections import deque
import sys
# 确保能找到 policy 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.scheduler import pick_node, Weights, estimate_eta

# --- 常量与归一化函数 (与训练脚本保持一致) ---
MINMAX = {
    "CPU": {"min": 1.0, "max": 5.0},
    "MEMORY": {"min": 2.0, "max": 16.0},
    "DELAY": {"min": 5.0, "max": 500.0},
    "LOAD": {"min": 0.0, "max": 1.0}
}
LOG_DELAY = False
DELAY_MIN_LOG = math.log1p(MINMAX["DELAY"]["min"])
DELAY_MAX_LOG = math.log1p(MINMAX["DELAY"]["max"])

def _norm_delay(d):
    if LOG_DELAY:
        z = (math.log1p(float(d)) - DELAY_MIN_LOG) / (DELAY_MAX_LOG - DELAY_MIN_LOG + 1e-8)
        return max(0., min(1., z))
    a, b = MINMAX["DELAY"]["min"], MINMAX["DELAY"]["max"]
    return (d - a) / (b - a + 1e-8)

def _denorm_delay(z):
    z = max(0., min(1., float(z)))
    if LOG_DELAY:
        return float(math.expm1(z * (DELAY_MAX_LOG - DELAY_MIN_LOG) + DELAY_MIN_LOG))
    a, b = MINMAX["DELAY"]["min"], MINMAX["DELAY"]["max"]
    return z * (b - a) + a

def norm4(v4):
    def _n(x,k):
        a, b = MINMAX[k]["min"], MINMAX[k]["max"]
        return (x - a) / (b - a + 1e-8)
    return np.array([_n(v4[0], "CPU"), _n(v4[1], "MEMORY"), _norm_delay(v4[2]), _n(v4[3], "LOAD")], dtype=np.float32)

def denorm4(z4):
    def _d(x,k):
        a, b = MINMAX[k]["min"], MINMAX[k]["max"]
        return x * (b - a) + a
    return np.array([_d(z4[0], "CPU"), _d(z4[1], "MEMORY"), _denorm_delay(z4[2]), _d(z4[3], "LOAD")], dtype=np.float32)

# --- 模型定义 (与训练脚本保持一致) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0) / d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]
    
class TS_Transformer(nn.Module):
    def __init__(self,input_dim=4,d_model=128,nhead=4,num_layers=3,dim_ff=256,dropout=0.1):
        super().__init__()
        self.in_proj=nn.Linear(input_dim,d_model)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_ff,dropout=dropout,batch_first=True)
        self.encoder=nn.TransformerEncoder(enc,num_layers=num_layers)
        self.pe=PositionalEncoding(d_model)
        self.head=nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Linear(d_model,4))
    def forward(self, x):
        h = self.in_proj(x)
        h = self.pe(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.head(last)
    
def read_csv_grouped(path):
    groups={}
    with open(path,"r",encoding="utf-8") as f:
        r=csv.reader(f)
        for row in r:
            if len(row)<5: 
                continue
            try:
                nid=int(row[0])
                vals=[float(row[1]), float(row[2]), float(row[3]), float(row[4])]
                groups.setdefault(nid,[]).append(vals)
            except: 
                continue
    return groups

def predict_one_step(model, device, hist):
    x_n = np.stack([norm4(x) for x in hist], axis=0)
    x_t = torch.from_numpy(x_n).unsqueeze(0).to(device)
    with torch.no_grad():
        y_n = model(x_t).cpu().numpy()[0]
    return denorm4(y_n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/node_logs.csv")
    ap.add_argument("--ckpt", default="model_output/ts_transformer_best.pt")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--steps", type=int, default=200, help="模拟请求数")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log_delay", action="store_true")
    ap.add_argument("--req_cost", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--margin_ms", type=float, default=14.0)
    ap.add_argument("--fair_gamma", type=float, default=1.0)
    ap.add_argument("--fair_scale_ms", type=float, default=25.0)
    ap.add_argument("--fair_window", type=int, default=20)
    ap.add_argument("--cooldown_k", type=int, default=2)
    ap.add_argument("--cooldown_ms", type=float, default=4.0)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--safety_threshold_ms", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--out_csv", default="results/sim_results.csv")
    args = ap.parse_args()

    global LOG_DELAY
    LOG_DELAY = args.log_delay
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu"

    groups = read_csv_grouped(args.source)
    node_ids = sorted(groups.keys())

    for nid in node_ids:
        if len(groups[nid]) < args.window + 1:
            raise ValueError(f"节点 {nid} 的数据不足: {len(groups[nid])} < {args.window + 1}")
        
    model = TS_Transformer(input_dim=4, d_model=args.d_model, nhead=args.nhead, num_layers=args.layers, dim_ff=args.ffn, dropout=args.dropout).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    hist = {nid: groups[nid][:args.window].copy() for nid in node_ids}
    t_ptr = {nid: args.window for nid in node_ids}

    w = Weights(alpha=args.alpha, beta=args.beta, margin_ms=args.margin_ms)

    # 初始化统计和均衡策略的状态
    eta_real_list = []
    assign_cnt = {nid: 0 for nid in node_ids}
    deadline_miss = 0
    deadline_ms = 80.0
    recent_assign = {nid: deque(maxlen=args.fair_window) for nid in node_ids}
    cooldown_left = {nid: 0 for nid in node_ids}

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "chosen_node", "eta_pred", "eta_real"])
        for step in range(1, args.steps + 1):
            preds = {nid: predict_one_step(model, device, hist[nid]) for nid in node_ids}

            # 计算包含均衡惩罚的最终分数
            final_scores = {}
            for nid, pred in preds.items():
                eta = estimate_eta(pred, req_cost=args.req_cost, w=w)
                
                ra = recent_assign[nid]
                recent_ratio = (sum(ra) / max(1, len(ra))) if len(ra) > 0 else 0.0
                fair_ms = args.fair_gamma * recent_ratio * args.fair_scale_ms
                cd_ms = args.cooldown_ms * cooldown_left[nid]
                
                final_scores[nid] = eta + fair_ms + cd_ms

            # *** 使用新的pick_node函数进行决策 ***
            best_node = pick_node(
                final_scores, 
                top_k=args.top_k, 
                epsilon=args.epsilon,
                threshold_ms=args.safety_threshold_ms
            )
            
            # 使用真实世界的下一步状态来评估和推进模拟
            real_eta = None
            for nid in node_ids:
                idx = min(t_ptr[nid], len(groups[nid]) - 1)
                real_next = groups[nid][idx]
                t_ptr[nid] += 1
                hist[nid].pop(0)
                hist[nid].append(real_next)
                if nid == best_node:
                    r_cpu, r_mem, r_delay, r_load = real_next
                    comp_real = args.req_cost / max(1e-3, r_cpu * (1.0 - r_load))
                    real_eta = args.alpha * r_delay + args.beta * comp_real

            # 更新均衡策略的状态
            for nid in node_ids:
                if cooldown_left[nid] > 0:
                    cooldown_left[nid] -= 1
            cooldown_left[best_node] = args.cooldown_k
            for nid in node_ids:
                recent_assign[nid].append(1 if nid == best_node else 0)
            
            # 更新统计数据
            assign_cnt[best_node] += 1
            eta_real_list.append(real_eta)
            if real_eta > deadline_ms:
                deadline_miss += 1

            wr.writerow([step, best_node, f"{final_scores[best_node]:.3f}", f"{real_eta:.3f}"])

    # --- 最终总结 ---
    def mean(xs): return float(np.mean(xs)) if xs else 0.0
    def p95(xs): return float(np.percentile(xs, 95)) if xs else 0.0
    miss_ratio = deadline_miss / max(1, args.steps)
    imbalance = np.std(list(assign_cnt.values()))

    print(f"[Sim Summary] steps={args.steps}, "
          f"avg_eta(real)={mean(eta_real_list):.2f} ms, "
          f"p95_eta(real)={p95(eta_real_list):.2f} ms, "
          f"miss_ratio={miss_ratio:.3f}, "
          f"imbalance_std={imbalance:.2f}")
    print(f"[Saved] {args.out_csv}")

if __name__ == "__main__":
    main()
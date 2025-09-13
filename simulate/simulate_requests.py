import os
import csv
import math
import argparse
import random
import numpy as np
import torch
from torch import nn
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from policy.scheduler import pick_node, Weights, estimate_eta

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
                vals=[
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4])
                ]
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
    ap.add_argument("--steps", type=int, default=200, help="Number of simulated requests")
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
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=0.2)
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
            raise ValueError(f"节点 {nid} 的长度不足，至少需要 {args.window + 1} 条")
        
    model = TS_Transformer(
        input_dim=4,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.ffn,
        dropout=args.dropout
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    hist = {nid: groups[nid][:args.window].copy() for nid in node_ids}
    t_ptr = {nid: args.window for nid in node_ids}

    w = Weights(
        alpha=args.alpha, 
        beta=args.beta, 
        margin_ms=args.margin_ms
    )

    eta_real_list = []
    assign_cnt = {nid: 0 for nid in node_ids}
    deadline_miss = 0
    deadline_ms = 80.0

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "step",
            "chosen_node",
            "eta_pred",
            "eta_real",
            "net_delay_pred",
            "comp_pred"
        ])
        for step in range(1, args.steps + 1):
            preds = {
                nid: predict_one_step(model, device, hist[nid]) for nid in node_ids
            }

            best, scores = pick_node(
                preds,
                req_cost=args.req_cost,
                w=w,
                top_k=args.top_k,
                epsilon=args.epsilon
            )

            b_pred = preds[best]
            cpu, mem, delay, load = b_pred
            eff = max(1e-3, cpu * max(0.0, 1.0 - load))
            comp_pred = args.req_cost / eff
            eta_pred = scores[best]

            real_eta = None
            for nid in node_ids:
                idx = min(t_ptr[nid], len(groups[nid]) - 1)
                real_next = groups[nid][idx]
                t_ptr[nid] = min(t_ptr[nid] + 1, len(groups[nid]) - 1)
                hist[nid].pop(0)
                hist[nid].append(real_next)
                if nid == best:
                    r_cpu, r_mem, r_delay, r_load = real_next
                    r_eff = max(1e-3, r_cpu * max(0.0, 1.0 - r_load))
                    comp_real = args.req_cost / r_eff
                    real_eta = args.alpha * r_delay + args.beta * comp_real

            assign_cnt[best] += 1
            eta_real_list.append(real_eta)
            if real_eta > deadline_ms:
                deadline_miss += 1

            wr.writerow([
                step, best, 
                f"{eta_pred:.3f}", f"{real_eta:.3f}", 
                f"{delay:.3f}", f"{comp_pred:.3f}"
            ])

    def mean(xs): 
        return float(np.mean(xs)) if xs else 0.0
    def p95(xs): 
        return float(np.percentile(xs, 95)) if xs else 0.0

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
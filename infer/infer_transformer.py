import os
import csv
import math
import argparse
import numpy as np
import torch
from torch import nn

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
    def _n(x, k):
        a, b = MINMAX[k]["min"], MINMAX[k]["max"]
        return (x - a) / (b - a + 1e-8)
    return np.array([
        _n(v4[0], "CPU"),
        _n(v4[1], "MEMORY"),
        _norm_delay(v4[2]),
        _n(v4[3], "LOAD")
    ], dtype=np.float32)

def denorm4(z4):
    def _d(x, k):
        a, b = MINMAX[k]["min"], MINMAX[k]["max"]
        return x * (b - a) + a
    return np.array([
        _d(z4[0], "CPU"),
        _d(z4[1], "MEMORY"),
        _denorm_delay(z4[2]),
        _d(z4[3], "LOAD")
    ], dtype=np.float32)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]
    
class TS_Transformer(nn.Module):
    def __init__(self, input_dim=4, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pe = PositionalEncoding(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))
    def forward(self, x):
        h = self.in_proj(x)
        h = self.pe(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        return self.head(last)
    
def read_csv(path, node_id):
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 5: 
                continue
            try:
                nid = int(row[0])
                if nid != node_id:
                    continue
                vals = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
                seq.append(vals)
            except:
                continue
    return seq

def read_txts(path, node_id):
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = [x.strip() for x in line.strip().split(",")]
            if len(p) < 5: 
                continue
            try:
                nid = int(p[0])
                if nid != node_id:
                    continue
                vals = [float(p[1]), float(p[2]), float(p[3]), float(p[4])]
                seq.append(vals)
            except:
                continue
    return seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model_output/ts_transformer_best.pt")
    ap.add_argument("--source", default="results/node_logs.csv", help="优先用CSV；若不存在再尝试TXT")
    ap.add_argument("--txt_fallback", default="results/node_logs.txt")
    ap.add_argument("--node_id", type=int, default=0)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--steps", type=int, default=1, help="滚动预测步数")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log_delay", action="store_true")
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    global LOG_DELAY
    LOG_DELAY = args.log_delay

    device = "cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu"

    if os.path.exists(args.source):
        seq = read_csv(args.source, args.node_id)
    elif os.path.exists(args.txt_fallback):
        seq = read_txts(args.txt_fallback, args.node_id)
    else:
        raise FileNotFoundError(f"Neither {args.source} nor {args.txt_fallback} exists.")
    
    if len(seq) < args.window:
        raise ValueError(f"Node {args.node_id} has only {len(seq)} records, less than window {args.window}.")
    
    hist = seq[-args.window:].copy()

    model = TS_Transformer(input_dim=4, d_model=args.d_model, nhead=args.nhead, num_layers=args.layers, dim_ff=args.ffn, dropout=args.dropout).to(device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"未找到权重 {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    outputs = []
    with torch.no_grad():
        for _ in range(args.steps):
            x_n = np.stack([norm4(x) for x in hist], axis=0)
            x_t = torch.from_numpy(x_n).unsqueeze(0).to(device)
            y_n = model(x_t).cpu().numpy()[0]
            y = denorm4(y_n).tolist()
            outputs.append(y)
            hist.pop(0)
            hist.append(y)

    for i, y in enumerate(outputs, 1):
        print(f"Step {i}: CPU={y[0]:.2f}, MEMORY={y[1]:.2f}, DELAY={y[2]:.2f}, LOAD={y[3]:.2f}")

if __name__ == "__main__":
    main()

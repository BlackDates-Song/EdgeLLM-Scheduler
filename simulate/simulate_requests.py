import os
import csv
import math
import argparse
import random
import numpy as np
import torch
from torch import nn
from policy.scheduler import pick_node, Weights

MINMAX = {
    "CPU": {"min": 1.0, "max": 5.0},
    "MEMORY": {"min": 2.0, "max": 16.0},
    "DELAY": {"min": 5.0, "max": 500.0},
    "LOAD": {"min": 0.0, "max": 1.0}
}
LOG_DELAY = True
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
    ap.add_argument("--source", default="results/node_logs.csv", help="优先用CSV；若不存在再尝试TXT")
    ap.add_argument("--ckpt", default="model_output/ts_transformer_best.pt")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100, help="模拟请求数")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--req_cost", type=float, default=1.0, help="每个请求的计算资源消耗")
    ap.add_argument("--alpha", type=float, default=1.0, help="网络延迟权重")
    ap.add_argument("--beta", type=float, default=1.0, help="计算时间权重")
    ap.add_argument("")
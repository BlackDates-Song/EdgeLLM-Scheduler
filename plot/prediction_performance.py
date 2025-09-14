import os
import math
import argparse
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from collections import defaultdict

MODEL_PATH = 'model_outputs/ts_transformer_best.pt'
LOG_DATA_PATH = 'results/node_logs.csv'
NODE_TO_PLOT = 0  # 要绘制的节点ID
SAMPLE_TO_PLOT = 100  # 要绘制的样本数量

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

def norm4(v):
    def _n(x, k):
        a, b = MINMAX[k]["min"], MINMAX[k]["max"]
        return (x - a) / (b - a + 1e-8)
    return np.array([_n(v[0],"CPU"), _n(v[1],"MEMORY"), _norm_delay(v[2]), _n(v[3],"LOAD")], dtype=np.float32)

def denorm4(z):
    def _d(u, k):
        a, b = MINMAX[k]["min"], MINMAX[k]["max"]
        return u * (b - a) + a
    return np.array([_d(z[0],"CPU"), _d(z[1],"MEMORY"), _denorm_delay(z[2]), _d(z[3],"LOAD")], dtype=np.float32)

def parse_line(line):
    p=[x.strip() for x in line.strip().split(",")]
    if len(p)<5: 
        return None
    try:
        nid=int(p[0]); vals=np.array([float(p[1]),float(p[2]),float(p[3]),float(p[4])],dtype=np.float32)
        return nid, vals
    except: 
        return None
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]
    
class TS_Transformer(nn.Module):
    def __init__(self, input_dim=4, num_nodes=10, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.node_embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=d_model)
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pe = PositionalEncoding(d_model)
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, 4))
    def forward(self, x, node_ids):
        node_emb = self.node_embedding(node_ids)
        h = self.in_proj(x); h = self.pe(h); h = self.encoder(h)
        last_hidden_state = h[:, -1, :]
        combined = torch.cat([last_hidden_state, node_emb], dim=1)
        return self.head(combined)
    
def main():
    parser = argparse.ArgumentParser("Plot prediction performance of TS Transformer")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="模型路径")
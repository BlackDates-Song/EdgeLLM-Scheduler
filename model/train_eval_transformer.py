import os
import json
import argparse
import math
import csv
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

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

def norm4(z):
    def _n(x, k):
        a, b = MINMAX[k]["min"], MINMAX[k]["max"]
        return (x - a) / (b - a + 1e-8)
    return np.array([_n(z[0],"CPU"), _n(z[1],"MEMORY"), _norm_delay(z[2]), _n(z[3],"LOAD")], dtype=np.float32)

def denorm4(z):
    def _d(x, k):
        a, b = MINMAX[k]["min"], MINMAX[k]["max"]
        return x * (b - a) + a
    return np.array([_d(z[0],"CPU"), _d(z[1],"MEMORY"), _denorm_delay(z[2]), _d(z[3],"LOAD")], dtype=np.float32)

def parse_line(line):
    p = [x.strip() for x in line.strip().split(",")]
    if len(p) < 5:
        return None
    try:
        nid = int(p[0])
        vals = np.array([float(p[1]),float(p[2]),float(p[3]),float(p[4])],dtype=np.float32)
        return nid, vals
    except:
        return None
    
def build_windows_per_node(series, window, node_idx, use_delta=False):
    out = []
    if len(series) <= window:
        return out
    arr = np.stack(series, axis=0)
    if use_delta:
        diff = np.zeros_like(arr)
        diff[1:] = arr[1:] - arr[:-1]
    for i in range(window, len(series)):
        hist_raw = arr[i-window:i]
        tgt_raw = arr[i]
        hist_n = np.stack([norm4(x) for x in hist_raw], axis=0)
        if use_delta:
            diff_win = diff[i-window:i]
            def n4d(d4):
                c = (d4[0]) / (MINMAX["CPU"]["max"] - MINMAX["CPU"]["min"] + 1e-8)
                m = (d4[1]) / (MINMAX["MEMORY"]["max"] - MINMAX["MEMORY"]["min"] + 1e-8)
                d = (d4[2]) / (MINMAX["DELAY"]["max"] - MINMAX["DELAY"]["min"] + 1e-8)
                l = (d4[3]) / (MINMAX["LOAD"]["max"] - MINMAX["LOAD"]["min"] + 1e-8)
                return np.clip(np.array([c,m,d,l],dtype=np.float32), -1.0, 1.0)
            diff_n = np.stack([n4d(x) for x in diff_win], axis=0)
            hist_n = np.concatenate([hist_n, diff_n], axis=1)
        tgt_n = norm4(tgt_raw)
        out.append((hist_n, tgt_n, hist_raw, tgt_raw, node_idx))
    return out

class SeqDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        Hn, Yn, Hr, Yr, NID = self.samples[idx]
        return torch.from_numpy(Hn), torch.from_numpy(Yn), torch.from_numpy(Hr), torch.from_numpy(Yr), NID

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]
    
class TS_Transformer(nn.Module):
    def __init__(self, input_dim=4, num_nodes=5, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.node_embedding = nn.Embedding(num_embeddings=num_nodes, embedding_dim=d_model)
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pe = PositionalEncoding(d_model)
        self.head = nn.Sequential(nn.Linear(d_model*2, d_model), nn.ReLU(), nn.Linear(d_model, 4))
    def forward(self, x, node_ids):
        node_emb = self.node_embedding(node_ids)
        h = self.in_proj(x)
        h = self.pe(h)
        h = self.encoder(h)
        last = h[:, -1, :]
        combined = torch.cat([last, node_emb], dim=1)
        return self.head(combined)

def mae_rmse(gt, pred):
    mae = np.mean(np.abs(pred - gt), axis=0)
    rmse = np.sqrt(np.mean((pred - gt) ** 2, axis=0))
    return mae, rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/node_logs.txt")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", default="results/pred_vs_gt_transformer.csv")
    parser.add_argument("--log_delay", action="store_true")
    parser.add_argument("--use_delta", action="store_true")
    parser.add_argument("--delay_weight", type=float, default=3.0, help="加权MSE中Delay的权重")
    parser.add_argument("--huber_delay", action="store_true", help="仅对 Delay 维使用 Huber(SmoothL1) 损失")
    parser.add_argument("--grad_clip", type=float, default=0.0, help=">0 时启用梯度裁剪（max_norm）")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--num_nodes", type=int, default=5, help="节点总数（用于Embedding）")
    args = parser.parse_args()

    global LOG_DELAY
    LOG_DELAY = args.log_delay
    print(f"[Info] LOG_DELAY = {LOG_DELAY}, USE_DELTA = {args.use_delta}, HUBER_DELAY = {args.huber_delay}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu"
    print(f"[Info] Using device: {device}")

    groups = defaultdict(list)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            p = parse_line(line)
            if p is None:
                continue
            nid, v = p
            groups[nid].append(v)

    node_id_map = {
        node_id: i for i, node_id in enumerate(sorted(groups.keys()))
    }

    train, val = [], []
    total = 0
    for nid, seq in groups.items():
        node_idx = node_id_map[nid]
        pairs = build_windows_per_node(seq, args.window, node_idx, use_delta=args.use_delta)
        total += len(pairs)
        if not pairs: 
            continue
        split = max(1, int((1.0 - args.val_ratio) * len(pairs)))
        train.extend(pairs[:split])
        val.extend(pairs[split:])
    print(f"[Data] total={total}, train={len(train)}, val={len(val)}, window={args.window}")

    tr_loader = DataLoader(SeqDataset(train), batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(SeqDataset(val), batch_size=args.batch_size, shuffle=False)

    in_dim = train[0][0].shape[1] if len(train) > 0 else (8 if args.use_delta else 4)
    model = TS_Transformer(input_dim=in_dim, num_nodes=args.num_nodes, d_model=args.d_model, nhead=args.nhead, num_layers=args.layers, dim_ff=args.ffn, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    base_weights = torch.tensor([1.0, 1.0, args.delay_weight, 1.0], device=device)
    mse_fn = nn.MSELoss(reduction="none")
    huber_fn = nn.SmoothL1Loss(reduction="none")

    best = float("inf")
    patience = 6
    bad = 0
    os.makedirs("model_output", exist_ok=True)
    ckpt_path = "model_output/ts_transformer_best.pt"
    for ep in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        for Hn, Yn, _, _, NIDs in tr_loader:
            Hn = Hn.to(device)
            Yn = Yn.to(device)
            NIDs = NIDs.to(device)
            pr = model(Hn, NIDs)
            if args.huber_delay:
                mse = mse_fn(pr, Yn)
                hub = huber_fn(pr, Yn)
                w_mse = torch.tensor([1.0, 1.0, 0.0, 1.0], device=device)
                w_hub = torch.tensor([0.0, 0.0, args.delay_weight, 0.0], device=device)
                loss = (mse * w_mse + hub * w_hub).mean()
            else:
                mse = mse_fn(pr, Yn)
                loss = (mse * base_weights).mean()
            opt.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
            tr_loss += loss.item() * Hn.size(0)
        tr_loss /= len(tr_loader.dataset)

        # ---- eval ----
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for Hn, Yn, _, _, NIDs in va_loader:
                Hn = Hn.to(device)
                Yn = Yn.to(device)
                NIDs = NIDs.to(device)
                pr = model(Hn, NIDs)
                if args.huber_delay:
                    mse = mse_fn(pr, Yn)
                    hub = huber_fn(pr, Yn)
                    w_mse = torch.tensor([1.0, 1.0, 0.0, 1.0], device=device)
                    w_hub = torch.tensor([0.0, 0.0, args.delay_weight, 0.0], device=device)
                    loss = (mse * w_mse + hub * w_hub).mean()
                else:
                    mse = mse_fn(pr, Yn)
                    loss = (mse * base_weights).mean()
                va_loss += loss.item() * Hn.size(0)
        va_loss /= (len(va_loader.dataset) if len(va_loader.dataset) > 0 else 1)
        print(f"[Epoch {ep:02d}] train_wmse={tr_loss:.6f}  val_wmse={va_loss:.6f}")

        if va_loss < best - 1e-6:
            best = va_loss
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStop] no improvement for {patience} epochs.")
                break
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

    gts, preds = [], []
    rows = []
    with torch.no_grad():
        for Hn, Yn, Hr, Yr, NIDs in va_loader:
            Hn = Hn.to(device)
            NIDs = NIDs.to(device)
            pr_n = model(Hn, NIDs).cpu().numpy()
            gt_n = Yn.numpy()
            pr = np.stack([denorm4(p) for p in pr_n], axis=0)
            gt = np.stack([denorm4(g) for g in gt_n], axis=0)
            gts.append(gt)
            preds.append(pr)
            for i in range(pr.shape[0]):
                rows.append([gt[i,0],gt[i,1],gt[i,2],gt[i,3],pr[i,0],pr[i,1],pr[i,2],pr[i,3]])
    if len(rows) > 0:
        gts = np.concatenate(gts,axis=0)
        preds=np.concatenate(preds,axis=0)
        mae, rmse = mae_rmse(gts, preds)
        dims=["CPU","Memory","Delay","Load"]
        for i,d in enumerate(dims):
            print(f"{d}: MAE = {mae[i]:.4f}, RMSE = {rmse[i]:.4f}")
        os.makedirs("results",exist_ok=True)
        with open(args.out_csv,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow(["gt_cpu","gt_mem","gt_delay","gt_load","pred_cpu","pred_mem","pred_delay","pred_load"])
            w.writerows(rows)
        print(f"[Saved] {args.out_csv}")
    else:
        print("[Warning] 验证集为空或过小，未输出 CSV。")

if __name__ == "__main__":
    main()
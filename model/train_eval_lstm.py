# train_eval_lstm.py
import os
import argparse
from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import csv
import random

# --------- 全局配置（可在命令行覆盖） ---------
MINMAX = {
    "CPU":   {"min": 1.0, "max": 5.0},
    "MEM":   {"min": 2.0, "max": 16.0},
    "DELAY": {"min": 5.0, "max": 500.0},
    "LOAD":  {"min": 0.0, "max": 1.0},
}

# 是否对 delay 使用对数尺度
LOG_DELAY = False  # 会被命令行参数覆盖

# 预计算 delay 的 log1p 边界
DELAY_MIN_LOG = math.log1p(MINMAX["DELAY"]["min"])  # log(1+5)
DELAY_MAX_LOG = math.log1p(MINMAX["DELAY"]["max"])  # log(1+500)

def _norm_delay(d):
    if LOG_DELAY:
        z = (math.log1p(float(d)) - DELAY_MIN_LOG) / (DELAY_MAX_LOG - DELAY_MIN_LOG + 1e-8)
        return max(0.0, min(1.0, z))
    # 走普通 min-max
    a, b = MINMAX["DELAY"]["min"], MINMAX["DELAY"]["max"]
    return (d - a) / (b - a + 1e-8)

def _denorm_delay(z):
    z = max(0.0, min(1.0, float(z)))
    if LOG_DELAY:
        val = math.expm1(z * (DELAY_MAX_LOG - DELAY_MIN_LOG) + DELAY_MIN_LOG)
        return float(val)
    a, b = MINMAX["DELAY"]["min"], MINMAX["DELAY"]["max"]
    return z * (b - a) + a

def norm4(v):
    def _n(x, k):
        a, b = MINMAX[k]["min"], MINMAX[k]["max"]
        return (x - a) / (b - a + 1e-8)
    return np.array([
        _n(v[0], "CPU"),
        _n(v[1], "MEM"),
        _norm_delay(v[2]),
        _n(v[3], "LOAD"),
    ], dtype=np.float32)

def denorm4(z):
    def _d(u, k):
        a, b = MINMAX[k]["min"], MINMAX[k]["max"]
        return u * (b - a) + a
    return np.array([
        _d(z[0], "CPU"),
        _d(z[1], "MEM"),
        _denorm_delay(z[2]),
        _d(z[3], "LOAD"),
    ], dtype=np.float32)

def parse_log_line(line):
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 5: return None
    try:
        nid = int(parts[0])
        cpu = float(parts[1]); mem = float(parts[2]); delay = float(parts[3]); load = float(parts[4])
        return nid, np.array([cpu, mem, delay, load], dtype=np.float32)
    except:
        return None

def build_windows_per_node(series, window):
    """
    series: list of 4-d vectors (raw scale) sorted by time
    return: list of (hist_norm[T,4], tgt_norm[4], hist_raw[T,4], tgt_raw[4])
    """
    out = []
    if len(series) <= window: return out
    # 构造滑窗
    for i in range(window, len(series)):
        hist_raw = np.stack(series[i-window:i], axis=0)   # [T,4]
        tgt_raw  = series[i]                               # [4]
        hist_norm = np.stack([norm4(x) for x in hist_raw], axis=0)
        tgt_norm  = norm4(tgt_raw)
        out.append((hist_norm, tgt_norm, hist_raw, tgt_raw))
    return out

class SeqDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        Hn, Yn, Hr, Yr = self.samples[idx]
        return torch.from_numpy(Hn), torch.from_numpy(Yn), torch.from_numpy(Hr), torch.from_numpy(Yr)

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=4, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, dropout=dropout if layers>1 else 0.0, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)
        )
    def forward(self, x):  # x: [B, T, 4]
        out, _ = self.lstm(x)          # [B, T, H]
        last = out[:, -1, :]           # 取最后时刻表示
        return self.head(last)         # [B, 4] (normalized targets)

def mae_rmse(gt, pred):
    mae = np.mean(np.abs(pred - gt), axis=0)
    rmse = np.sqrt(np.mean((pred - gt)**2, axis=0))
    return mae, rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/node_logs.txt", help="原始日志：node_id,cpu,mem,delay,load")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.2, help="每个node按时间后20%做验证")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="results/pred_vs_gt_lstm.csv")
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--log_delay", action="store_true", help="对 delay 使用 log1p 归一化")
    args = ap.parse_args()

    global LOG_DELAY
    LOG_DELAY = args.log_delay
    print(f"[Info] LOG_DELAY = {LOG_DELAY}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = "cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu"
    print(f"[Info] Using device: {device}")

    # 1) 读取并按 node 分组
    groups = defaultdict(list)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed is None: continue
            nid, vec4 = parsed
            groups[nid].append(vec4)

    # 2) 按节点构造滑窗，并做“时间切分”（避免泄漏）
    train_samples, val_samples = [], []
    total_pairs = 0
    for nid, series in groups.items():
        series = series  # 假定文件本身已按时间；如需可在此排序
        pairs = build_windows_per_node(series, args.window)
        total_pairs += len(pairs)
        if not pairs: continue
        # 时间切分：每个 node 后段做验证
        split = max(1, int((1.0 - args.val_ratio) * len(pairs)))
        train_samples.extend(pairs[:split])
        val_samples.extend(pairs[split:])
    print(f"[Data] total_pairs={total_pairs}, train={len(train_samples)}, val={len(val_samples)}, window={args.window}")

    train_ds = SeqDataset(train_samples)
    val_ds   = SeqDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 3) 构建模型
    model = LSTMRegressor(input_dim=4, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # 4) 训练
    best_val = float("inf")
    patience, bad = 5, 0  # 简易 early stopping
    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for Hn, Yn, _, _ in train_loader:
            Hn = Hn.to(device)      # [B,T,4], normalized
            Yn = Yn.to(device)      # [B,4], normalized
            pred_n = model(Hn)      # [B,4]
            loss = loss_fn(pred_n, Yn)
            optim.zero_grad(); loss.backward(); optim.step()
            tr_loss += loss.item() * Hn.size(0)
        tr_loss /= len(train_loader.dataset)

        # 验证（用 MSE 监控）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Hn, Yn, _, _ in val_loader:
                Hn = Hn.to(device); Yn = Yn.to(device)
                pred_n = model(Hn)
                loss = loss_fn(pred_n, Yn)
                val_loss += loss.item() * Hn.size(0)
        val_loss /= (len(val_loader.dataset) if len(val_loader.dataset)>0 else 1)
        print(f"[Epoch {epoch:02d}] train_mse={tr_loss:.6f}  val_mse={val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), "model_output/lstm_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStop] no improvement for {patience} epochs.")
                break

    # 5) 加载最佳权重并在验证集上评估（反归一化）
    if os.path.exists("data/lstm_best.pt"):
        model.load_state_dict(torch.load("data/lstm_best.pt", map_location=device))
        model.eval()

    gts_all, preds_all = [], []
    rows = []
    with torch.no_grad():
        for Hn, Yn, Hr, Yr in val_loader:
            Hn = Hn.to(device)
            pred_n = model(Hn).cpu().numpy()     # normalized
            gt_n   = Yn.numpy()

            # 反归一化
            pred = np.stack([denorm4(p) for p in pred_n], axis=0)
            gt   = np.stack([denorm4(g) for g in gt_n], axis=0)

            gts_all.append(gt); preds_all.append(pred)

            # 写出对照（取 batch 的原始历史最后一步也可存一下，这里只存 target 对比）
            for i in range(pred.shape[0]):
                rows.append([
                    gt[i,0], gt[i,1], gt[i,2], gt[i,3],
                    pred[i,0], pred[i,1], pred[i,2], pred[i,3]
                ])

    if len(rows)==0:
        print("[Warn] 验证集太小或分割为空，无法评估。")
        return

    gts_all   = np.concatenate(gts_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)

    mae, rmse = mae_rmse(gts_all, preds_all)
    dims = ["CPU","Memory","Delay","Load"]
    for i, d in enumerate(dims):
        print(f"{d}: MAE = {mae[i]:.4f}, RMSE = {rmse[i]:.4f}")

    # 6) 保存对照 CSV
    os.makedirs("data", exist_ok=True)
    out_csv = args.out_csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_cpu","gt_mem","gt_delay","gt_load","pred_cpu","pred_mem","pred_delay","pred_load"])
        w.writerows(rows)
    print(f"[Saved] {out_csv}")
    print("[Done] LSTM baseline finished.")
    
if __name__ == "__main__":
    main()

import os, re, csv, math, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = "model_output"
DATA_FILE = "data/training_data_cleaned.txt"
SCALE_JSON = "data/scale.json"
OUT_CSV = "data/pred_vs_gt_multi.csv"
PLOT_ERR = "data/multi_error_hist.png"
PLOT_LINE = "data/multi_pred_vs_gt.png"

_PAIR = re.compile(r'(CPU|MEM|DELAY|LOAD)\s*=\s*([-+]?\d+(?:\.\d+)?)', re.I)
_NUM = re.compile(r'[-+]?\d+(?:\.\d+)?')

LIMITS = [(1.0, 5.0), (2.0, 16.0), (5.0, 500.0), (0.0, 1.0)]

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})
    END_ID = tokenizer.convert_tokens_to_ids("<END>")
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    return tokenizer, model, device, END_ID

def load_scale():
    with open(SCALE_JSON, "r", encoding="utf-8") as f:
        scale = json.load(f)
    return scale

def extract_first_four(pred_text: str):
    txt = pred_text.replace('"', ' ')
    txt = txt.split("<END>")[0].split("->")[0]
    got = {}
    for k, v in _PAIR.findall(txt):
        got[k.upper()] = float(v)
    order = ["CPU", "MEM", "DELAY", "LOAD"]
    if all(k in got for k in order):
        return [got[k] for k in order]
    nums = _NUM.findall(txt)
    if len(nums) >= 4:
        try:
            return [float(x) for x in nums[:4]]
        except:
            return None
    return None

def parse_gt_line(tgt: str):
    return extract_first_four(tgt)

def predict_next_state(history_str, tokenizer, model, device, END_ID):
    prompt = history_str if history_str.strip().endswith("->") else (history_str.strip() + " ->")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            no_repeat_ngram_size=3,
            eos_token_id=END_ID,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    after = decoded.split("->", 1)[1] if "->" in decoded else decoded
    return extract_first_four(after)

def read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().strip('"')
            if not line or "->" not in line:
                continue
            hist, tgt = line.split("->", 1)
            pairs.append((hist.strip(), tgt.strip()))
    return pairs

def denorm4(vals_norm, scale):
    c = vals_norm[0] * scale["CPU"]
    m = vals_norm[1] * scale["MEM"]
    d = vals_norm[2] * scale["DELAY"]
    l = vals_norm[3] * scale["LOAD"]
    return [c, m, d, l]

def clamp4(xs):
    out = []
    for v,(lo,hi) in zip(xs, LIMITS):
        if v != v:
            v = lo
        v = min(max(v, lo), hi)
        out.append(v)
    return out

def mae_rmse(gt_arr, pred_arr):
    metrics = []
    for j in range(gt_arr.shape[1]):
        gt = gt_arr[:, j]
        pr = pred_arr[:, j]
        diff = pr - gt
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        metrics.append((mae, rmse))
    return metrics

def plot_error_hist(all_errors, out_path):
    plt.figure(figsize=(7, 4))
    plt.hist(all_errors, bins=40)
    plt.title("Error Distribution (all dimensions)")
    plt.xlabel("Prediction - Ground Truth")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_delay_curve(gt_delay, pred_delay, out_path, N=120):
    n = min(N, len(gt_delay))
    x = np.arange(n)
    plt.figure(figsize=(10, 4))
    plt.plot(x, gt_delay[:n], label="GT Delay")
    plt.plot(x, pred_delay[:n], label="Pred Delay")
    plt.title(f"Prediction vs GT (Delay) - first {n} samples")
    plt.xlabel("Sample")
    plt.ylabel("Delay (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    tokenizer, model, device, END_ID = load_model()
    scale = load_scale()
    pairs = read_pairs(DATA_FILE)
    print(f"Loaded {len(pairs)} samples from {DATA_FILE}")

    kept = 0
    drop_gt = drop_pred = 0
    preds, gts, rows = [], [], []

    for hist, tgt in pairs:
        gt_n = parse_gt_line(tgt)
        if gt_n is None or len(gt_n) != 4:
            drop_gt += 1
            continue
        pr_n = predict_next_state(hist, tokenizer, model, device, END_ID)
        if pr_n is None or len(pr_n) != 4:
            drop_pred += 1
            continue
        gt = clamp4(denorm4(gt_n, scale))
        pr = clamp4(denorm4(pr_n, scale))
        gts.append(gt)
        preds.append(pr)
        rows.append((hist, *gt, *pr))
        kept += 1

    print(f"Kept {kept} samples, dropped {drop_gt} GT and {drop_pred} Pred")
    if kept == 0:
        print("No valid samples to evaluate.")
        return

    gts = np.array(gts, dtype=float)
    preds = np.array(preds, dtype=float)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["history_input",
                    "gt_cpu", "gt_mem", "gt_delay", "gt_load",
                    "pred_cpu", "pred_mem", "pred_delay", "pred_load"
        ])
        w.writerows(rows)
    print(f"Saved comparison CSV to {OUT_CSV}")

    dims = ["CPU", "Memory", "Delay", "Load"]
    metrics = mae_rmse(gts, preds)
    for i, (mae, rmse) in enumerate(metrics):
        print(f"{dims[i]}: MAE = {mae:.4f}, RMSE = {rmse:.4f}")

    all_err = (preds - gts).reshape(-1)
    plot_error_hist(all_err, PLOT_ERR)
    print(f"Saved error histogram to {PLOT_ERR}")

    plot_delay_curve(gts[:, 2], preds[:, 2], PLOT_LINE)
    print(f"Saved delay curve to {PLOT_LINE}")

if __name__ == "__main__":
    main()

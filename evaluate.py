import os, re, csv, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = "model_output"
DATA_FILE = "data/training_data.txt"
OUT_CSV = "data/pred_vs_gt_multi.csv"
PLOT_ERR = "data/multi_error_hist.png"
PLOT_LINE = "data/multi_pred_vs_gt.png"

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    return tokenizer, model, device

_float_pat = re.compile(r'[-+]?\d+(?:\.\d+)?')

def extract_first_four(pred_text: str):
    after = pred_text.split("->",1)[1] if "->" in pred_text else pred_text
    after = after.replace("", '').split("->")[0]
    nums = _float_pat.findall(after)
    return nums[:4]

def str4_to_floats(s: str):
    nums = _float_pat.findall(s)
    return [float(x) for x in nums[:4]] if len(nums) >= 4 else [float("nan")]*4

def predict_next_state(history_str, tokenizer, model, device):
    prompt = history_str.strip()
    if not prompt.endswitch("->"):
        prompt = prompt + " ->"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    vals = extract_first_four(decoded)
    if len(vals) < 4:
        return [float("nan")]*4
    return [float(x) for x in vals]

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

def mae_rmse(gt_arr, pred_arr):
    metrics = []
    for j in range(gt_arr.shape[1]):
        gt = gt_arr[:, j]
        pr = pred_arr[:, j]
        mask = ~np.isnan(gt) & ~np.isnan(pr)
        if mask.sum() == 0:
            metrics.append((float("nan"), float("nan")))
            continue
        diff = pr[mask] - gt[mask]
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
    plt.ylabel("Delay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    tokenizer, model, device = load_model()
    pairs = read_pairs(DATA_FILE)
    print(f"Loaded {len(pairs)} samples from {DATA_FILE}")

    preds, gts, rows = [], [], []
    for hist, tgt in pairs:
        gt4 = str4_to_floats(tgt)
        pr4 = predict_next_state(hist, tokenizer, model, device)
        gts.append(gt4)
        preds.append(pr4)
        rows.append((hist, *pr4, *gt4))

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
    all_err = all_err[~np.isnan(all_err)]
    plot_error_hist(all_err, PLOT_ERR)
    print(f"Saved error histogram to {PLOT_ERR}")

    plot_delay_curve(gts[:, 2], preds[:, 2], PLOT_LINE)
    print(f"Saved delay curve to {PLOT_LINE}")

if __name__ == "__main__":
    main()

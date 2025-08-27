import os
import argparse
from collections import defaultdict
import json
import random

DEFAULT_SCALE = {
    "CPU": {"min": 1.0, "max": 5.0},
    "MEM": {"min": 2.0, "max": 16.0},
    "DELAY": {"min": 5.0, "max": 500.0},
    "LOAD": {"min": 0.0, "max": 1.0}
}

def parse_line(line: str):
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 5:
        return None
    try:
        node_id = int(parts[0])
        cpu = float(parts[1])
        mem = float(parts[2])
        delay = float(parts[3])
        load = float(parts[4])
        return node_id, [cpu, mem, delay, load]
    except:
        return None

def norm4(v4, scale=DEFAULT_SCALE):
    def _n(x,k):
        a, b = scale[k]["min"], scale[k]["max"]
        if b <= a:
            return 0.0
        z = (x - a) / (b - a)
        return max(0.0, min(1.0, z))
    c, m, d, l = v4
    return [_n(c, "CPU"), _n(m, "MEM"), _n(d, "DELAY"), _n(l, "LOAD")]

def denorm4(v4, scale=DEFAULT_SCALE):
    def _d(x,k):
        a, b = scale[k]["min"], scale[k]["max"]
        return x * (b - a) + a
    c, m, d, l = v4
    return [_d(c, "CPU"), _d(m, "MEM"), _d(d, "DELAY"), _d(l, "LOAD")]

def fmt4(v4):
    c,m,d,l = v4
    return f"CPU={c:.6f},MEM={m:.6f},DELAY={d:.6f},LOAD={l:.6f}"

def build_windows(series, W):
    out = []
    if len(series) <= W:
        return out
    for i in range(W, len(series)):
        hist = series[i-W:i]
        tgt = series[i]
        hist_n = [x for step in hist for x in norm4(step)]
        out.append((hist_n, tgt))
    return out

def smooth_tgt(tgt):
    c, m, d, l = tgt
    if abs(l - 0.0) < 1e-8:
        l = 0.01
    if abs(l - 1.0) < 1e-8:
        l = 0.99
    if d < 10.0:
        d = max(5.0, min(500.0, d + random.uniform(-2.0, 2.0)))
    return [c, m, d, l]

def oversample(pairs, scale=DEFAULT_SCALE, downsample_extreme_load=True):
    boosted = []
    for hist_n, tgt_n in pairs:
        tgt_sm = smooth_tgt(tgt_n)
        _, _, d_ms, l_raw = tgt_sm
        times = 1
        if d_ms > 300:
            times = 8
        elif 100 <= d_ms <= 300:
            times = 6
        if 0.3 <= l_raw <= 0.7:
            times = min(8, times + 2)
        if downsample_extreme_load and (abs(l_raw - 0.0) < 1e-8 or abs(l_raw - 1.0) < 1e-8):
            if random.random() < 0.7:
                continue
        for _ in range(times):
            boosted.append((hist_n, tgt_n))
    return boosted

def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="result/node_logs.csv", help="原始日志文件,每行为: node_id,cpu,mem,delay,load")
    ap.add_argument("--output", type=str, default="result/training_data_cleaned.csv", help="处理后的输出文件(带<END>)")
    ap.add_argument("--preview", default="result/training_data_preview.csv", help="预览输出文件地址")
    ap.add_argument("--scale", default="data/scale.json", help="归一化比例文件(JSON)，默认值为CPU=5.0,MEM=16.0,DELAY=500.0,LOAD=1.0")
    ap.add_argument("--window", type=int, default=10, help="历史窗口长度")
    ap.add_argument("--shuffle", action="store_true", help="是否打乱数据")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--preview_rows", type=int, default=300, help="预览输出的行数")
    if args is not None:
        args = ap.parse_args(args)
    else:
        args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    groups = defaultdict(list)
    n_raw = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            nid, vec4 = parsed
            groups[nid].append(vec4)
            n_raw += 1

    all_pairs = []
    for nid, seq in groups.items():
        all_pairs.extend(build_windows(seq, args.window))

    boosted = oversample(all_pairs, DEFAULT_SCALE, downsample_extreme_load=True)

    if args.shuffle:
        random.shuffle(boosted)

    with open(args.output, "w", encoding="utf-8") as f:
        for hist_n, tgt_sm in boosted:
            tgt_n = norm4(tgt_sm)
            steps = [hist_n[i:i+4] for i in range(0, len(hist_n), 4)]
            history_str = " ; ".join(fmt4(s) for s in steps)
            target_str = fmt4(tgt_n)
            f.write(f"{history_str} -> {target_str} <END>\n")

    with open(args.scale, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_SCALE, f, ensure_ascii=False, indent=2)

    try:
        import csv
        k = min(args.preview_rows, len(boosted))
        sample = boosted[:k]
        with open(args.preview, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["idx"]
            for t in range(args.window):
                header += [f"h{t+1}_cpu","h{t+1}_mem","h{t+1}_delay","h{t+1}_load"]
            header += ["y_cpu","y_mem","y_delay","y_load"]
            w.writerow(header)
            for i, (hist_n, tgt_n) in enumerate(sample):
                steps = [hist_n[j:j+4] for j in range(0, len(hist_n), 4)]
                row = [i]
                for s in steps:
                    row += [f"{v:.4f}" for v in denorm4(s)]
                row += [f"{v:.4f}" for v in tgt_n]
                w.writerow(row)
    except Exception as e:
        print(f"[preview] skip writing preview due to: {e}")

    print(f"raw_lines={n_raw}, nodes={len(groups)},"
          f" windows={len(all_pairs)}, boosted={len(boosted)}, window_size={args.window},"
          f" output='{args.output}', preview='{args.preview}', scale='{args.scale}'")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
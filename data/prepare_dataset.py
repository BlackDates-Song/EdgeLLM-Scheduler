import os
import argparse
from collections import defaultdict
import json

DEFAULT_SCALE = {
    "CPU": 5.0,
    "MEM": 16.0,
    "DELAY": 500.0,
    "LOAD": 1.0
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
    except Exception:
        return None

def norm4(v4, scale):
    c, m, d, l = v4
    return [c/scale["CPU"], m/scale["MEM"], d/scale["DELAY"], l/scale["LOAD"]]

def denorm4(v4, scale):
    c, m, d, l = v4
    return [c*scale["CPU"], m*scale["MEM"], d*scale["DELAY"], l*scale["LOAD"]]

def fmt4(v4):
    c,m,d,l = v4
    return f"CPU={c:.2f},MEM={m:.2f},DELAY={d:.2f},LOAD={l:.2f}"

def build_windows(series, W):
    out = []
    if len(series) <= W:
        return out
    for i in range(W, len(series)):
        hist = series[i-W:i]
        tgt = series[i]
        # hist_flat = []
        # for step in hist:
        #     hist_flat.extend(step)
        # hist_str = ",".join(f"{x:.2f}" for x in hist_flat)
        # tgt_str = fmt4(tgt)
        # out.append((hist_str, tgt_str))
        out.append((hist, tgt))
    return out

def oversample(line_pairs, delay_thr=0.35, load_thr=0.70, factor=3):
    boosted = []
    for h, t in line_pairs:
        d = t[2]
        l = t[3]
        times = factor if (d > delay_thr or l > load_thr) else 1
        for _ in range(times):
            boosted.append((h, t))
    return boosted

def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/node_logs.csv", help="原始日志文件,每行为: node_id,cpu,mem,delay,load")
    ap.add_argument("--output", type=str, default="data/training_data_cleaned.csv", help="处理后的输出文件(带<END>)")
    ap.add_argument("--scale", default="data/scale.json", help="归一化比例文件(JSON)，默认值为CPU=5.0,MEM=16.0,DELAY=500.0,LOAD=1.0")
    ap.add_argument("--window", type=int, default=10, help="历史窗口长度")
    ap.add_argument("--delay_thr", type=float, default=70.0, help="过采样阈值：延迟(ms)")
    ap.add_argument("--load_thr", type=float, default=0.70, help="过采样阈值：负载(rate)")
    ap.add_argument("--factor", type=int, default=3, help="高难样本重复写入次数")
    ap.add_argument("--shuffle", action="store_true", help="是否打乱数据")
    if args is not None:
        args = ap.parse_args(args)
    else:
        args = ap.parse_args()

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

    scale = DEFAULT_SCALE
    json.dump(scale, open(args.scale, "w"))

    all_pairs = []
    for nid, seq in groups.items():
        seq_norm = [norm4(v4, scale) for v4 in seq]
        all_pairs.extend(build_windows(seq_norm, args.window))

    boosted = oversample(all_pairs, delay_thr=args.delay_thr, load_thr=args.load_thr, factor=args.factor)

    if args.shuffle:
        import random
        random.shuffle(boosted)

    with open(args.output, "w", encoding="utf-8") as f:
        for hist, tgt in boosted:
            history_str = " ; ".join(fmt4(h) for h in hist)
            target_str = fmt4(tgt)
            f.write(f"{history_str} -> {target_str} <END>\n")
    
    print(f"raw_lines={n_raw}, nodes={len(groups)},"
          f" windows={len(all_pairs)}, boosted={len(boosted)},"
          f" window_size={args.window}, output='{args.output}', scale='{args.scale}'")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
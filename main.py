import argparse
import subprocess
import sys
import os

PY = sys.executable

def run(cmd_list):
    print("[CMD]", " ".join(cmd_list))
    p = subprocess.run(cmd_list)
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    
def stage_gen_logs(args):
    cmd = [
        PY, "data/generate_logs.py",
        "--out_txt", str(args.out_txt),
        "--out_csv", str(args.out_csv),
        "--num_nodes", str(args.num_nodes),
        "--steps", str(args.steps),
        "--interval", str(args.interval)
    ]
    run(cmd)

def stage_train_lstm(args):
    cmd = [
        PY, "model/train_eval_lstm.py",
        "--input", str(args.input),
        "--window", str(args.window),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--hidden", str(args.hidden),
        "--layers", str(args.layers),
    ]
    if args.log_delay:
        cmd.append("--log_delay")
    if args.no_cuda:
        cmd.append("--no_cuda")
    run(cmd)

def stage_train_transformer(args):
    cmd = [
        PY, "model/train_eval_transformer.py",
        "--input", str(args.input),
        "--window", str(args.window),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--d_model", str(args.d_model),
        "--nhead", str(args.nhead),
        "--layers", str(args.t_layers),
        "--ffn", str(args.ffn),
        "--dropout", str(args.dropout),
        "--delay_weight", str(args.delay_weight),
    ]
    if args.log_delay:
        cmd.append("--log_delay")
    if args.use_delta:
        cmd.append("--use_delta")
    if args.no_cuda:
        cmd.append("--no_cuda")
    run(cmd)

def main():
    ap = argparse.ArgumentParser("EdgeLLM Scheduler - unified entry")
    ap.add_argument("--stage", required=True, choices=["gen_logs", "train_lstm", "train_transformer"])

    # ------ gen_logs 参数 ------
    ap.add_argument("--out_txt", default="results/node_logs.txt")
    ap.add_argument("--out_csv", default="results/node_logs.csv")
    ap.add_argument("--num_nodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--interval", type=float, default=0.05)
    ap.add_argument("--cpu_min", type=float)
    ap.add_argument("--cpu_max", type=float)
    ap.add_argument("--mem_min", type=float)
    ap.add_argument("--mem_max", type=float)
    ap.add_argument("--delay_min", type=float)
    ap.add_argument("--delay_max", type=float)
    ap.add_argument("--load_min", type=float)
    ap.add_argument("--load_max", type=float)
    # ------ 模型通用参数 ------
    ap.add_argument("--input", default="results/node_logs.txt", help="日志文件路径（默认用 results/node_logs.txt）")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--log_delay", action="store_true")
    ap.add_argument("--no_cuda", action="store_true")
    # ------ LSTM 参数 ------
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    # ------ Transformer 参数 ------
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--t_layers", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_delta", action="store_true")
    ap.add_argument("--delay_weight", type=float, default=3.0)
    args = ap.parse_args()

    if args.stage == "gen_logs":
        stage_gen_logs(args)
    elif args.stage == "train_lstm":
        stage_train_lstm(args)
    elif args.stage == "train_transformer":
        stage_train_transformer(args)

if __name__ == "__main__":
    main()

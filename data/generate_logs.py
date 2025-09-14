import os
import time
import csv
import argparse
import sys
import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from edge.edge_node import EdgeNode



def generate_logs(
    out_txt="results/node_logs.txt",
    out_csv="results/node_logs.csv",
    num_nodes=5,
    steps=200,
    interval=0.05,
    seed=42
):
    random.seed(seed)

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    nodes = [EdgeNode(i) for i in range(num_nodes)]

    # generate logs
    with open(out_txt, "w") as log_file, open(out_csv, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        for t in range(steps):
            for node in nodes:
                node.update()
                log_line = node.to_log()
                log_file.write(log_line + "\n")

                # write to csv
                row_values = log_line.strip().split(",")[:5]
                csv_writer.writerow(row_values)

            time.sleep(interval)

    print(f"Logs saved to:\n  -> {out_txt} (raw)\n  -> {out_csv} (csv)")

def main():
    ap = argparse.ArgumentParser("Generate synthetic node logs")
    ap.add_argument("--out_txt", default="results/node_logs.txt")
    ap.add_argument("--out_csv", default="results/node_logs.csv")
    ap.add_argument("--num_nodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--interval", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    generate_logs(
        out_txt=args.out_txt,
        out_csv=args.out_csv,
        num_nodes=args.num_nodes,
        steps=args.steps,
        interval=args.interval,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
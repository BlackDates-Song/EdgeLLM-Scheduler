import os
import time
import csv
from edge.edge_node import EdgeNode

def generate_logs():
    # Generate multiple nodes
    NUM_NODES = 5
    NODES = [EdgeNode(i) for i in range(NUM_NODES)]

    # output path
    os.makedirs("data", exist_ok=True)
    log_path = "result/node_logs.txt"
    csv_path = "result/node_logs.csv"

    # generate logs
    with open(log_path, "w") as log_file, open(csv_path, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        for t in range(200):
            for node in NODES:
                node.update()
                log_line = node.to_log()
                log_file.write(log_line + "\n")

                # write to csv
                row_values = log_line.strip().split(",")[:5]
                csv_writer.writerow(row_values)

            time.sleep(0.05)  # Sleep for 0.05 seconds before the next update

    print(f"Logs saved to:\n  -> {log_path} (raw)\n  -> {csv_path} (for inference)")
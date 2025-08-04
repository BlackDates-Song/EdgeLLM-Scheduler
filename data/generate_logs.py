import os
import time
from edge.edge_node import EdgeNode

def generate_logs():
    # Generate multiple nodes
    NUM_NODES = 5
    NODES = [EdgeNode(i) for i in range(NUM_NODES)]

    # output path
    os.makedirs("data", exist_ok=True)
    log_path = "data/node_logs.txt"

    # generate logs
    with open(log_path, "w") as f:
        for t in range(200):  # simulate 200 time steps
            for node in NODES:
                node.update()
                log_line = node.to_log()
                f.write(log_line + "\n")
            time.sleep(0.05)  # simulate time delay between updates

    print(f"Logs generated and saved to {log_path}")
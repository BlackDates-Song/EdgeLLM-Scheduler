import os
from transformers import GPT2Tokenizer
from datasets import Dataset
import csv

INPUT_FILE = "data/node_logs.csv"
OUTPUT_FILE = "data/training_data.csv"
SEQ_LEN = 5

def prepare_dataset():
    with open(INPUT_FILE, 'r', newline="") as f:
        reader = csv.reader(f)
        data = list(reader)

    # data格式：[node_id, cpu, memory, delay, load]
    node_data = {}
    for row in data:
        node_id = row[0]
        values = list(map(float, row[1:]))
        if node_id not in node_data:
            node_data[node_id] = []
        node_data[node_id].append(values)

    # 生成训练数据
    training_rows = []
    for node_id, records in node_data.items():
        for i in range(len(records) - SEQ_LEN):
            hist = []
            for j in range(SEQ_LEN):
                hist.extend(records[i + j])
            target = records[i + SEQ_LEN]
            input_str = ",".join(map(str, hist))
            output_str = ",".join(map(str, target))
            training_rows.append([input_str +" -> "+ output_str])

    # 保存为CSV文件
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in training_rows:
            writer.writerow(row)

    print(f"Training data saved to {OUTPUT_FILE}, total {len(training_rows)} rows.")

if __name__ == "__main__":
    prepare_dataset()

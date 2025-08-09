import os
from transformers import GPT2Tokenizer
from datasets import Dataset
'''
    单变量预测
'''
'''
# Load the log file
def load_logs(log_file):
    with open(log_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

# Encode the log lines using the tokenizer
def tokenize_data(lines, tokenizer):
    examples = tokenizer(lines, truncation=True, padding=False)
    return {"input_ids": examples["input_ids"]}

# main function
def prepare_dataset():
    log_path = "data/node_logs.txt"
    assert os.path.exists(log_path), f"Log file {log_path} does not exist."

    #load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    #load log file
    lines = load_logs(log_path)
    print(f"Loaded {len(lines)} lines from the log file.")

    #build dataset
    ds = Dataset.from_dict(tokenize_data(lines, tokenizer))
    ds.save_to_disk("data/tokenized_dataset")

    print(f"Dataset saved to 'data/tokenized_dataset' with {len(ds)} examples.")
'''

'''
    多变量预测
'''
def load_logs(log_file):
    with open(log_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def prepare_dataset():
    log_path = "data/node_logs.txt"
    lines = load_logs(log_path)
    print(f"Loaded {len(lines)} lines from the log file.")

    # 按节点分组：用字典，key 是节点编号，value 是该节点所有行数据
    node_dict = {}
    for line in lines:
        parts = line.split(",")
        node_id = parts[0]
        values = parts[1:]
        if node_id not in node_dict:
            node_dict[node_id] = []
        node_dict[node_id].append(values)

    dataset_pairs = []
    for node_id, records in node_dict.items():
        for i in range(len(records) - 1):
            curr_values = ",".join(records[i])
            next_values = ",".join(records[i+1])
            dataset_pairs.append(f"{curr_values} -> {next_values}")

    print(f"Total training pairs: {len(dataset_pairs)}")

    dataset = Dataset.from_dict({"text": dataset_pairs})

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    os.makedirs("data/tokenized_dataset", exist_ok=True)
    tokenized_dataset.save_to_disk("data/tokenized_dataset")
    print(f"Tokenized dataset saved to 'data/tokenized_dataset' with {len(tokenized_dataset)} examples.")

if __name__ == "__main__":
    prepare_dataset()
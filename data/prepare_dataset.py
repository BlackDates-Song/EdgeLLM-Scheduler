import os
from transformers import GPT2Tokenizer
from datasets import Dataset

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


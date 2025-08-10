import torch
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset

class MultiVarDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    
        self.examples = []
        for line in lines:
            if "->" not in line:
                continue
            encoded = tokenizer(
                line, 
                truncation=True, 
                max_length=block_size, 
                padding='max_length'
            )
            self.examples.append(encoded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = {key: torch.tensor(val) for key, val in self.examples[i].items()}
        return item
    
def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
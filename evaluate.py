import os, re, csv, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = "model_output"
DATA_FILE = "data/training_data.txt"
OUT_CSV = "data/pred_vs_gt_multi.csv"
PLOT_ERR = "data/multi_error_hist.png"
PLOT_LINE = "data/multi_pred_vs_gt.png"

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    return tokenizer, model, device

_float_pat = re.compile(r'[-+]?\d+(?:\.\d+)?')

def 

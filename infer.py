import torch
from transformers import GPT2LMHeadModel
from data.tokenize_dataset import get_tokenizer
import re
import json

MODEL_DIR = "model_output"
MAX_LENGTH = 50
SCALE = json.load(open("data/scale.json", 'r'))

_num = re.compile(r'[-+]?\d+(?:\.\d+)?')
_pair = re.compile(r'(CPU|MEM|DELAY|LOAD)\s*=\s*([-+]?\d+(?:\.\d+)?)', re.I)


def load_model():
    tokenizer = get_tokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})
    END_ID = tokenizer.convert_tokens_to_ids("<END>")
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device, END_ID

def denorm4(v4):
    return [
        v4[0] * SCALE["CPU"],
        v4[1] * SCALE["MEM"],
        v4[2] * SCALE["DELAY"],
        v4[3] * SCALE["LOAD"],
    ]

def get_first_prediction(text:str):
    text = text.split("<END>")[0].split("->")[1] if "->" in text else text
    text = text.replace('"', ' ').strip()

    got = {}
    for k, v in _pair.findall(text):
        got[k.upper()] = float(v)

    order = ["CPU", "MEM", "DELAY", "LOAD"]
    vals= []
    for k in order:
        if k in got:
            vals.append(got[k])
        else:
            nums = _num.findall(text)
            if len(nums) >= 4:
                vals = [float(n) for n in nums[:4]]
                break
    if len(vals) != 4:
        return None
    return vals

def predict_next_state(prompt, tokenizer, model, device, END_ID):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(inputs['input_ids'][0]) + MAX_LENGTH,
            num_return_sequences=1,
            do_sample=False,
            no_repeat_ngram_size=3,
            eos_token_id=END_ID,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    first_prediction = get_first_prediction(decoded)
    vals = denorm4(first_prediction)

    limits = [(1.0, 5.0), (2.0, 16.0), (5.0, 500.0), (0.0, 1.0)]
    vals = [min(max(v, lo), hi) for v, (lo, hi) in zip(vals, limits)]
    print("Predicted Next State ->", f"{vals[0]:.2f}, {vals[1]:.2f}, {vals[2]:.0f}, {vals[3]:.2f}")

def format_prediction(prediction):
    nums = []
    for x in prediction:
        try:
            pred = float(x)
        except:
            pred = float("nan")
        nums.append(pred)

    fmt = [".2f", ".2f", ".2f", ".2f"]
    return ",".join(f"{pred:{f}}" for pred, f in zip(nums, fmt))

def main():
    tokenizer, model, device, END_ID = load_model()
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("> Input: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if "->" not in user_input:
            prompt = user_input + " ->"
        else:
            prompt = user_input

        prediction = predict_next_state(prompt, tokenizer, model, device, END_ID)
        print(f"\nRaw Prediction: {prediction}\n")

        if len(prediction) == 4:
            print("Predicted Next State ->", format_prediction(prediction), "\n")
        else:
            print("只解析到", len(prediction), "个数：", prediction)

if __name__ == "__main__":
    main()
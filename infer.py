import torch
from transformers import GPT2LMHeadModel
from data.tokenize_dataset import get_tokenizer
import re

MODEL_DIR = "model_output"
MAX_LENGTH = 50

def load_model():
    tokenizer = get_tokenizer()
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

def get_first_prediction(prediction:str):
    after = prediction.split("->", 1)[1] if "->" in prediction else prediction
    after = after.replace('"', ' ').split("->")[0]
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', after)
    return nums[:4]

def predict_next_state(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(inputs['input_ids'][0]) + MAX_LENGTH,
            num_return_sequences=1,
            do_sample=False
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    first_prediction = get_first_prediction(decoded)
    return first_prediction

def main():
    tokenizer, model, device = load_model()
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("> Input: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if "->" not in user_input:
            prompt = user_input + " ->"
        else:
            prompt = user_input

        prediction = predict_next_state(prompt, tokenizer, model, device)
        print(f"\nRaw Prediction: {prediction}\n")

        if len(prediction) == 4:
            print("Predicted Next State ->", ",".join(prediction))
        else:
            print("只解析到", len(prediction), "个数：", prediction)

if __name__ == "__main__":
    main()
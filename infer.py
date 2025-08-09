import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import re

MODEL_PATH = "checkpoints/gpt2-lora-final"

'''
    单变量预测
'''
'''
def load_model():
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device

def predict_next_state(input_str, model, tokenizer, device):
    prompt = input_str.strip() + "->"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 10,  # Generate up to 10 tokens
            num_beams=5,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #get the lastest part
    generated_part = output_text[len(prompt):].strip()
    cleaned = re.findall(r"[0-9]+(?:\.[0-9]+)?", generated_part)
    if cleaned:
        return cleaned[0]  # Return the first predicted value
    else:
        print("No valid prediction found.")
        return "0.0"

def run_inference():
    model, tokenizer, device = load_model()

    print("Model loaded successfully. You can start typing your input.")
    print("Type 'exit' or 'quit' to stop the inference loop.")

    while True:
        try:
            user_input = input("\n> Input: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            prediction = predict_next_state(user_input, model, tokenizer, device)
            print(f"Prediction: {prediction}")
        except KeyboardInterrupt:
            print("\nExiting inference loop.")
            break
'''

"""
    多变量预测
"""
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

def predict_next_state(input_str, model, tokenizer, device):
    prompt = f"{input_str.strip()} ->"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "->" in prediction:
        prediction = prediction.split("->")[1].strip()

    numbers = re.findall(r"\d+\.\d+", prediction)

    while len(numbers) < 4:
        numbers.append("0.0")
    return numbers[:4]

def run_inference():
    model, tokenizer, device = load_model()

    print("Model loaded successfully. You can start typing your input.")
    print("Type 'exit' or 'quit' to stop the inference loop.")

    while True:
        try:
            user_input = input("\n> Input: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            pred = predict_next_state(user_input, model, tokenizer, device)
            print(f"Predicted next state -> : CPU: {pred[0]}, Memory: {pred[1]}, Delay: {pred[2]}, Load: {pred[3]}")
        except KeyboardInterrupt:
            print("\nExiting inference loop.")
            break
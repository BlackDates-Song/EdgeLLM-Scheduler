from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Input the node status sample
sample = "0,2.93,5.73,62.84,0.45"

# Encode the input sample
inputs = tokenizer(sample, return_tensors="pt")

# Generate predictions
outputs = model.generate(
    inputs["input_ids"],
    max_length=20,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
)

# Decode the generated output
print("Inputs:", sample)
print("Outputs:", tokenizer.decode(outputs[0], skip_special_tokens=True))
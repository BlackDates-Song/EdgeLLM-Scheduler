import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from data.tokenize_dataset import MultiVarDataset, get_tokenizer
from peft import LoraConfig, get_peft_model, TaskType

# INITIAL SETTING
DATA_FILE = "data/training_data_cleaned.txt"
OUTPUT_DIR = "model_output"
BLOCK_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 12
LR = 1e-5

LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["c_attn", "c_proj"]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #load tokenizer
    tokenizer = get_tokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": ["<END>"]})
    END_ID = tokenizer.convert_tokens_to_ids("<END>")

    #load dataset
    dataset = MultiVarDataset(DATA_FILE, tokenizer, block_size=BLOCK_SIZE)
    
    #split train
    train_size = int (0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    #load model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    #training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        learning_rate=LR,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True
    )

    #data collator
    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
    )

    #Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

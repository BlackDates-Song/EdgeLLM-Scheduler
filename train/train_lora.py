import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from data.tokenize_dataset import MultiVarDataset, get_tokenizer

# INITIAL SETTING
DATA_FILE = "data/training_data.txt"
OUTPUT_DIR = "model_output"
BLOCK_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 5
LR = 5e-5

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #load tokenizer
    tokenizer = get_tokenizer()

    #load dataset
    dataset = MultiVarDataset(DATA_FILE, tokenizer, block_size=BLOCK_SIZE)
    
    #split train
    train_size = int (0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    #load model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    #training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        learning_rate=LR,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"
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

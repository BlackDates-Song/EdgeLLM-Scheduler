import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType

def train_lora():
    # load pretrained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # use LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # load dataset
    dataset = load_from_disk("data/tokenized_dataset")

    # auto padding and MLM off
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints/gpt2_lora",
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=5e-5,
        save_strategy="epoch",
        logging_steps=20,
        fp16=True,
        report_to="none"
    )

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # start training
    print("Starting training...")
    trainer.train()
    trainer.save_model("checkpoints/gpt2-lora-final")
    print("Training completed. The model is saved in 'checkpoints/gpt2-lora-final'.")

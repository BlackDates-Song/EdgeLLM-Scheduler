import argparse

from data.generate_logs import generate_logs
from data.prepare_dataset import main as prepare_dataset
from train.train_lora import main as train_lora
from evaluate.infer import main as run_inference
from evaluate.batch_infer import run_batch_inference
from evaluate.evaluate import main as evaluate_main

def main():
    parser = argparse.ArgumentParser(description="EdgeLLM Scheduler Pipeline")
    parser.add_argument("--stage", type=str, required=True, choices=["generate", "train", "infer", "evaluate"], help="执行的阶段：generate, train, infer, evaluate")

    args, unknown = parser.parse_known_args()

    if args.stage == "generate":
        print("[Stage] Generating logs...")
        generate_logs()
    elif args.stage == "train":
        print("[Stage] Training model...")
        train_lora()
    elif args.stage == "evaluate":
        print("[Stage] Evaluating results...")
        evaluate_main()
    else:
        print(f"Unknown stage: {args.stage}")

if __name__ == "__main__":
    main()
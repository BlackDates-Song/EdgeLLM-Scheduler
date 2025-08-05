import argparse

from data.generate_logs import generate_logs
from data.prepare_dataset import prepare_dataset
from train.train_lora import train_lora
from infer import run_inference
from batch_infer import run_batch_inference

def main():
    parser = argparse.ArgumentParser(description="EdgeLLM Scheduler Pipeline")
    parser.add_argument("--stage", type=str, required=True, choices=["generate", "preprocess", "train", "infer", "batch_infer"], help="执行的阶段：generate, preprocess, train, infer, batch_infer")

    args = parser.parse_args()

    if args.stage == "generate":
        print("[Stage] Generating logs...")
        generate_logs()
    elif args.stage == "preprocess":
        print("[Stage] Preparing dataset...")
        prepare_dataset()
    elif args.stage == "train":
        print("[Stage] Training model...")
        train_lora()
    elif args.stage == "infer":
        print("[Stage] Running inference...")
        run_inference()
    elif args.stage == "batch_infer":
        print("[Stage] Running batch inference...")
        run_batch_inference(
            input_file="data/test_logs.csv",
            output_file="data/predicted_logs.csv"
        )
    else:
        print(f"Unknown stage: {args.stage}")

if __name__ == "__main__":
    main()
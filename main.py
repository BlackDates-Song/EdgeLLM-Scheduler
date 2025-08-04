import os
import sys

from data.generate_logs import generate_logs
from data.prepare_dataset import prepare_dataset

if __name__ == "__main__":
    # print("Starting EdgeLLM Scheduler...")
    # generate_logs()
    # print("Logs generated successfully.")
    prepare_dataset()
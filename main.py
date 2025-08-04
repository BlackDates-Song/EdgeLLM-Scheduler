import os
import sys

from data.generate_logs import generate_logs

if __name__ == "__main__":
    print("Starting EdgeLLM Scheduler...")
    generate_logs()
    print("Logs generated successfully.")
# EdgeLLM-Scheduler
This project is to build a simulation system that includes multiple edge nodes (comparable to nodes in a blockchain), and use lightweight large language models (such as DistilBERT or GPT2-small) to predict the status of each node, and intelligently schedule tasks to the nodes with the most suitable resources.

## Vitual Environment:
    Create a venv and active:
        python -m venv venv
        venv\Scripts\activate

## Quick Start (Mainline)
1) Prepare logs: 
   run sth and get `data/node_logs.txt` with rows `node_id,cpu,mem,delay,load`
2) LSTM:
   python train/train_eval_lstm.py --input data/node_logs.txt --window 20 --epochs 40 --batch_size 64 --hidden 256 --layers 2 --log_delay
3) Transformer:
   python train/train_eval_transformer.py --input data/node_logs.txt --window 30 --epochs 40 --batch_size 64 --d_model 128 --nhead 4 --layers 3 --ffn 256 --log_delay --use_delta
4) Results:
   CSVs under data/, best weights lstm_best.pt / ts_transformer_best.pt.(will move to other place somewhen)

Legacy (GPT-2, archived)
See legacy_gpt2/ (not maintained).

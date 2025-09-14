# EdgeLLM-Scheduler
This project is to build a simulation system that includes multiple edge nodes (comparable to nodes in a blockchain), and use lightweight large language models (such as DistilBERT or GPT2-small) to predict the status of each node, and intelligently schedule tasks to the nodes with the most suitable resources.

## Vitual Environment:
    Create a venv and active:
        python -m venv venv
        venv\Scripts\activate

## Quick Start (Mainline)
1) Prepare logs: 
   run `python main.py --stage gen_logs --num_nodes 10 --steps 1000 --interval 0.02 --seed 42` 
   get `results/node_logs.txt` with rows `node_id,cpu,mem,delay,load`

2) Transformer:
   `python main.py --stage train --input results/node_logs.txt --window 30 --epochs 40 --batch_size 64 --d_model 128 --nhead 4 --layers 3 --ffn 256 --dropout 0.1 --log_delay --delay_weight 5.0 --huber_delay --num_nodes 5`

3) Results:
   CSVs under `results/`, best weights under `model_output/`
      `ts_transformer_best.pt`.

4) Infer:
   One step infer (node 0):
      `python main.py --stage infer --node_id 0 --window 30 --steps_ahead 1 --log_delay`
   Multi-steps infer (node 3, 5 steps):
      `python main.py --stage infer --node_id 3 --window 30 --steps_ahead 5 --log_delay`

5) Simulate:
   `python main.py --stage simulate --source results/node_logs.csv --ckpt model_output/ts_transformer_best.pt --window 30 --steps 200 --req_cost 1.5 --margin_ms 14 --fair_gamma 1.0 --fair_scale_ms 25 --fair_window 20 --cooldown_k 2 --cooldown_ms 4 --top_k 2 --epsilon 0.2 --safety_threshold_ms 5.0`

## Result
For the current dataset and model architecture, any scheduling strategy that attempts to enforce load balancing will lead to a significant decline in service quality (latency and timeout rate), and it is impossible to reduce the load imbalance to an ideal level.

## Future Improvement
   - More complex models: Introduce Graph Neural Networks (GNN) to model the network topology relationships among nodes. 
   - More advanced algorithms: Use multi-objective reinforcement learning (Multi-Objective RL) to directly learn a scheduling strategy that can balance multiple objectives. 
   - Larger-scale data: Validate the model's generalization ability on more realistic and larger-scale datasets.

### Legacy (GPT-2, archived)
   See legacy_gpt2/ (not maintained).

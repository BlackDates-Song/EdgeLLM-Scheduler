import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

def load_ground_truth(log_path, target_index=3):
    """
    Load next value from node_logs.txt
    """
    with open(log_path, 'r') as f:
        lines = [line.strip().split(',') for line in f if line.strip()]

    # get sequence from each node
    node_data = {}
    for row in lines:
        node_id = int(row[0])
        if node_id not in node_data:
            node_data[node_id] = []
        node_data[node_id].append(float(row[target_index]))

    # mix each node's data as the ground truth
    ground_truth = []
    for node_id, seq in node_data.items():
        ground_truth.extend(seq[1:])
    
    return ground_truth

def load_predictions(pred_path):
    """
    Load predictions from predicted_logs.csv.
    """
    predictions = []
    with open(pred_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            try:
                pred = float(row[-1])
                predictions.append(pred)
            except:
                continue
    return predictions

def evaluate(ground_truth, predictions):
    """
    output evaluation metrics: RMSE and MAE
    """
    min_len = min(len(predictions), len(ground_truth))
    gt = np.array(ground_truth[:min_len])
    pred = np.array(predictions[:min_len])

    mae = mean_absolute_error(gt, pred)
    rmse = math.sqrt(mean_squared_error(gt, pred))

    print(f"\nEvaluation Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return gt, pred

def plot_results(gt, pred):
    """
    Plot the ground truth vs predictions.
    """
    errors = pred - gt

    # plot the histogram
    plt.figure(figsize=(10, 4))
    plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
    plt.title("误差分布图（预测值-真实值）")
    plt.xlabel("误差")
    plt.ylabel("数量")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/error_histogram.png")
    print("Error histogram saved as data/error_histogram.png")
    plt.show()

    # plot the predictions vs ground truth
    plt.figure(figsize=(12, 4))
    plt.plot(gt[:100], label="真实值", marker='o')
    plt.plot(pred[:100], label="预测值", marker='x')
    plt.title("前100个样本： 真实值 vs 预测值")
    plt.xlabel("样本编号")
    plt.ylabel("Delay (ms)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/prediction_vs_truth.png")
    print("Prediction vs Ground Truth plot saved as data/prediction_vs_truth.png")
    plt.show()

def main():
    log_path = "data/node_logs.txt"
    pred_path = "data/predicted_logs.csv"

    ground_truth = load_ground_truth(log_path)
    predictions = load_predictions(pred_path)

    gt, pred = evaluate(ground_truth, predictions)
    plot_results(gt, pred)

if __name__ == "__main__":
    main()
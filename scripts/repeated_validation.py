"""使用多个随机种子的重复分层验证检验固定最终模型的稳定性。

本脚本只读取原始训练集，不读取测试集，也不重复执行参数搜索。
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "results" / "final"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from src.data_loader import load_training_data
from src.experiment_utils import MODEL_NAMES, predict_original_labels, train_all_final_models


RANDOM_SEEDS = [23, 42, 123, 2024, 2025]
VALIDATION_RATIO = 0.2
METRICS = ("accuracy", "precision", "recall", "f1")


def save_csv(rows, summary_rows):
    with open(RESULTS_DIR / "Repeated Validation Results.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open(RESULTS_DIR / "Repeated Validation Summary.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_f1_stability(summary_rows):
    """用均值和标准差误差棒展示各模型验证集 F1 的稳定性。"""
    names = [row["model"] for row in summary_rows]
    means = [float(row["f1_mean"]) for row in summary_rows]
    stds = [float(row["f1_std"]) for row in summary_rows]
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(names, means, yerr=stds, capsize=6, color="#4C78A8")
    axis.set_ylim(max(0.9, min(means) - max(stds) * 3 - 0.005), 1.0)
    axis.set_ylabel("Validation F1-score")
    axis.set_title("Repeated Stratified Validation: F1 Mean and Standard Deviation")
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    for bar, mean, std in zip(bars, means, stds):
        axis.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.0005,
                  f"{mean:.4f} ± {std:.4f}", ha="center", va="bottom", fontsize=9)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "Repeated Validation F1 Stability.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    X_all, y_all = load_training_data()
    rows = []
    metric_values = defaultdict(lambda: defaultdict(list))
    for seed in RANDOM_SEEDS:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=VALIDATION_RATIO, random_state=seed,
            stratify=y_all, shuffle=True,
        )
        print(f"\nRandom seed: {seed}")
        for model_name, model in train_all_final_models(X_train, y_train).items():
            prediction = predict_original_labels(model_name, model, X_val)
            row = {
                "seed": seed, "model": model_name,
                "accuracy": accuracy_score(y_val, prediction),
                "precision": precision_score(y_val, prediction, pos_label=9),
                "recall": recall_score(y_val, prediction, pos_label=9),
                "f1": f1_score(y_val, prediction, pos_label=9),
            }
            rows.append(row)
            for metric in METRICS:
                metric_values[model_name][metric].append(row[metric])
            print(f"{model_name}: validation_f1={row['f1']:.6f}")

    summary_rows = []
    for model_name in MODEL_NAMES:
        summary = {"model": model_name, "n_seeds": len(RANDOM_SEEDS)}
        for metric in METRICS:
            values = np.asarray(metric_values[model_name][metric], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
        summary_rows.append(summary)
        print(f"{model_name}: F1={summary['f1_mean']:.6f} ± {summary['f1_std']:.6f} (n={len(RANDOM_SEEDS)})")
    save_csv(rows, summary_rows)
    plot_f1_stability(summary_rows)


if __name__ == "__main__":
    main()

"""按模型独立执行验证集网格搜索，避免重复训练无关模型。

示例：
    python run_grid_search.py --model ce
    python run_grid_search.py --model ce --ce-final-check
    python run_grid_search.py --model hinge
    python run_grid_search.py --model linear_svm
    python run_grid_search.py --model rbf_svm
"""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "results" / "tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from src.data_loader import load_data
from src.hyperparameter_search import search_ce_grid, search_hinge_grid
from src.svm_models import search_linear_svm_grid, search_rbf_svm_grid


# Cross-Entropy 的上一轮最优学习率位于 1.0 的右边界，
# 因此向更大的学习率、以及更小的正则化系数进行局部扩展。
CE_BOUNDARY_LEARNING_RATES = [0.8, 1.0, 1.2, 1.5]
CE_BOUNDARY_REG_LAMBDAS = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001]

# 边界搜索表明 learning_rate=1.5 仍为最优点，以下设置仅用于确认
# 更大学习率是否继续提升；固定正则化系数以避免无意义的重复训练。
CE_FINAL_LEARNING_RATES = [1.5, 1.8, 2.0]
CE_FINAL_REG_LAMBDAS = [0.00001]

HINGE_LEARNING_RATES = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
HINGE_REG_LAMBDAS = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.01]
SVM_C_VALUES = [0.01, 0.1, 1, 10, 100]
RBF_GAMMA_VALUES = ["scale", 0.001, 0.01, 0.1, 1]


def save_manual_result(result, model_name, output_path):
    """保存手写模型的二维搜索结果。"""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "learning_rate", "reg_lambda", "validation_f1"])
        for lr_index, learning_rate in enumerate(result["learning_rates"]):
            for reg_index, reg_lambda in enumerate(result["reg_lambdas"]):
                writer.writerow(
                    [
                        model_name,
                        learning_rate,
                        reg_lambda,
                        result["f1_matrix"][lr_index, reg_index],
                    ]
                )


def plot_manual_heatmap(result, title, output_path):
    """绘制学习率与正则化系数的 F1 热力图，并标出最优组合。"""
    matrix = result["f1_matrix"]
    fig, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto")
    axis.set_xticks(range(len(result["reg_lambdas"])))
    axis.set_xticklabels([f"{value:g}" for value in result["reg_lambdas"]])
    axis.set_yticks(range(len(result["learning_rates"])))
    axis.set_yticklabels([f"{value:g}" for value in result["learning_rates"]])
    axis.set_xlabel("L2 regularization coefficient")
    axis.set_ylabel("Learning rate")
    axis.set_title(title)

    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            axis.text(column, row, f"{value:.4f}", ha="center", va="center", fontsize=8)

    best_row = result["learning_rates"].index(result["best_learning_rate"])
    best_column = result["reg_lambdas"].index(result["best_reg_lambda"])
    axis.add_patch(
        plt.Rectangle((best_column - 0.5, best_row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2)
    )
    fig.colorbar(image, ax=axis, label="Validation F1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def save_svm_result(result, model_name, output_path, has_gamma=False):
    """保存 sklearn SVM 的一维或二维搜索结果。"""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "C", "gamma", "validation_f1"])
        if not has_gamma:
            for c_value, f1_value in zip(result["c_values"], result["f1_scores"]):
                writer.writerow([model_name, c_value, "", f1_value])
            return

        for c_index, c_value in enumerate(result["c_values"]):
            for gamma_index, gamma_value in enumerate(result["gamma_values"]):
                writer.writerow(
                    [model_name, c_value, gamma_value, result["f1_matrix"][c_index, gamma_index]]
                )


def main():
    parser = argparse.ArgumentParser(description="按模型独立运行验证集网格搜索")
    parser.add_argument(
        "--model",
        choices=["ce", "hinge", "linear_svm", "rbf_svm"],
        required=True,
        help="本次需要搜索的模型；每次仅训练指定模型",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="手写模型每组参数的训练轮数")
    parser.add_argument(
        "--ce-final-check",
        action="store_true",
        help="仅适用于 ce：固定 reg_lambda=1e-5，确认学习率 1.5、1.8、2.0",
    )
    args = parser.parse_args()

    if args.ce_final_check and args.model != "ce":
        parser.error("--ce-final-check 仅可与 --model ce 一起使用")

    X_train, X_val, _, y_train, y_val, _ = load_data()

    if args.model == "ce":
        if args.ce_final_check:
            learning_rates = CE_FINAL_LEARNING_RATES
            reg_lambdas = CE_FINAL_REG_LAMBDAS
            csv_path = RESULTS_DIR / "Cross Entropy Final Check.csv"
            figure_path = RESULTS_DIR / "Cross Entropy Final Check.png"
            figure_title = "Cross-Entropy Final Learning-rate Check"
        else:
            learning_rates = CE_BOUNDARY_LEARNING_RATES
            reg_lambdas = CE_BOUNDARY_REG_LAMBDAS
            csv_path = RESULTS_DIR / "Cross Entropy Boundary Search.csv"
            figure_path = RESULTS_DIR / "Cross Entropy Boundary Search.png"
            figure_title = "Cross-Entropy Boundary Grid Search"

        result = search_ce_grid(
            X_train, np.where(y_train == 9, 1, 0),
            X_val, np.where(y_val == 9, 1, 0),
            learning_rates, reg_lambdas, args.epochs,
        )
        save_manual_result(result, "Cross-Entropy", csv_path)
        plot_manual_heatmap(result, figure_title, figure_path)
        print(
            f"\nBest Cross-Entropy parameters: learning_rate={result['best_learning_rate']}, "
            f"reg_lambda={result['best_reg_lambda']}, validation_f1={result['best_f1']:.6f}"
        )
    elif args.model == "hinge":
        result = search_hinge_grid(
            X_train, np.where(y_train == 9, 1, -1),
            X_val, np.where(y_val == 9, 1, -1),
            HINGE_LEARNING_RATES, HINGE_REG_LAMBDAS, args.epochs,
        )
        save_manual_result(result, "Hinge Loss", RESULTS_DIR / "Hinge Loss Grid Search.csv")
        plot_manual_heatmap(result, "Hinge Loss Grid Search", RESULTS_DIR / "Hinge Loss Grid Search.png")
        print(
            f"\nBest Hinge Loss parameters: learning_rate={result['best_learning_rate']}, "
            f"reg_lambda={result['best_reg_lambda']}, validation_f1={result['best_f1']:.6f}"
        )
    elif args.model == "linear_svm":
        result = search_linear_svm_grid(X_train, y_train, X_val, y_val, SVM_C_VALUES)
        save_svm_result(result, "Linear SVM", RESULTS_DIR / "Linear SVM Grid Search.csv")
        print(f"\nBest Linear SVM parameters: C={result['best_c']}, validation_f1={result['best_f1']:.6f}")
    else:
        result = search_rbf_svm_grid(X_train, y_train, X_val, y_val, SVM_C_VALUES, RBF_GAMMA_VALUES)
        save_svm_result(result, "RBF SVM", RESULTS_DIR / "RBF SVM Grid Search.csv", has_gamma=True)
        print(
            f"\nBest RBF SVM parameters: C={result['best_c']}, gamma={result['best_gamma']}, "
            f"validation_f1={result['best_f1']:.6f}"
        )


if __name__ == "__main__":
    main()

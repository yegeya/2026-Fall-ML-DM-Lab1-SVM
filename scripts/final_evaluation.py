"""使用已确定的最优参数训练最终模型，并生成结果分析图表。

本脚本不执行网格搜索。它完成以下工作：
1. 在训练/验证划分上记录完整与局部放大的损失曲线；
2. 合并训练集和验证集，训练四个最终模型；
3. 记录训练、预测耗时和测试集指标；
4. 保存各模型的测试集错分样本图。
"""

import csv
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "results" / "final"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from src.ce_model import CE_Classifier
from src.data_loader import load_data
from src.hinge_model import HingeClassifier
from src.experiment_utils import (
    EPOCHS as SHARED_EPOCHS,
    FINAL_PARAMETERS as SHARED_FINAL_PARAMETERS,
    MODEL_NAMES,
    predict_original_labels,
    train_final_model,
)


EPOCHS = 1000
ZOOM_START_EPOCH = 50
ERROR_SAMPLES_PER_ROW = 6

# 所有参数均已通过训练集内部的验证集选择，测试集仅用于最终一次评估。
FINAL_PARAMETERS = {
    "Linear SVM": {"C": 0.1},
    "RBF SVM": {"C": 10, "gamma": "scale"},
    "Hinge Loss": {"learning_rate": 0.4, "reg_lambda": 0.00003},
    "Cross-Entropy": {"learning_rate": 1.5, "reg_lambda": 0.00001},
}


# 最终参数的唯一有效来源为公共模块；本地旧定义保留仅为兼容历史文件结构。
EPOCHS = SHARED_EPOCHS
FINAL_PARAMETERS = SHARED_FINAL_PARAMETERS


def plot_loss_curves(hinge_model, ce_model):
    """同时展示完整损失曲线和后期收敛区间，避免首轮损失掩盖细节。"""
    histories = [
        ("Hinge Loss", hinge_model.loss_history, hinge_model.val_loss_history),
        ("Cross-Entropy", ce_model.loss_history, ce_model.val_loss_history),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row, (name, train_loss, val_loss) in enumerate(histories):
        epochs = np.arange(1, len(train_loss) + 1)
        for axis, start_epoch, suffix in (
            (axes[row, 0], 1, "Full Training Process"),
            (axes[row, 1], ZOOM_START_EPOCH, f"Epoch {ZOOM_START_EPOCH}-{EPOCHS}"),
        ):
            mask = epochs >= start_epoch
            axis.plot(epochs[mask], np.asarray(train_loss)[mask], label="Training loss")
            axis.plot(epochs[mask], np.asarray(val_loss)[mask], label="Validation loss")
            axis.set_title(f"{name}: {suffix}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.grid(True, alpha=0.3)
            axis.legend()

    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "Loss Curves Full and Zoomed.png", dpi=300, bbox_inches="tight")
    plt.show()


def select_error_indices(y_true, y_pred):
    """返回全部错分样本，并按 8→9、9→8 的顺序分组。"""
    eight_as_nine = np.flatnonzero((y_true == 8) & (y_pred == 9))
    nine_as_eight = np.flatnonzero((y_true == 9) & (y_pred == 8))
    return np.concatenate([eight_as_nine, nine_as_eight]).astype(np.intp)


def plot_misclassified_samples(model_name, X_test, y_true, y_pred):
    """保存一个模型的全部错分手写数字图像。"""
    indices = select_error_indices(y_true, y_pred)
    n_columns = ERROR_SAMPLES_PER_ROW
    n_rows = max(1, (len(indices) + n_columns - 1) // n_columns)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(12, max(2.4, n_rows * 2.0)))
    axes = axes.ravel()

    for axis, index in zip(axes, indices):
        axis.imshow(X_test[index].reshape(28, 28), cmap="gray")
        axis.set_title(f"True: {y_true[index]}, Pred: {y_pred[index]}", fontsize=9)
        axis.axis("off")
    for axis in axes[len(indices):]:
        axis.axis("off")

    errors_8_as_9 = int(np.sum((y_true == 8) & (y_pred == 9)))
    errors_9_as_8 = int(np.sum((y_true == 9) & (y_pred == 8)))
    figure.suptitle(
        f"All Misclassified Test Samples: {model_name} "
        f"(8→9: {errors_8_as_9}, 9→8: {errors_9_as_8})",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    filename = f"Misclassified Samples {model_name.replace('-', ' ')}.png"
    figure.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()


def save_final_results(rows):
    """保存测试集指标和时间统计，便于直接插入实验报告表格。"""
    with open(RESULTS_DIR / "Final Test Results and Timing.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_training_time(rows):
    """以对数纵轴绘制耗时，保留毫秒级预测时间的可读性。"""
    names = [row["model"] for row in rows]
    train_seconds = [float(row["training_seconds"]) for row in rows]
    predict_seconds = [float(row["prediction_seconds"]) for row in rows]
    x = np.arange(len(names))
    width = 0.36

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, train_seconds, width, label="Training")
    prediction_bars = axis.bar(x + width / 2, predict_seconds, width, label="Prediction")
    axis.set_xticks(x, names)
    axis.set_ylabel("Time (seconds)")
    axis.set_title("Final Model Training and Prediction Time")
    axis.set_yscale("log")
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    axis.legend()
    # 训练耗时可由柱高直接比较；仅标注相差较大的预测耗时，避免与标题重叠。
    axis.bar_label(prediction_bars, fmt="%.4g s", padding=3, rotation=90, fontsize=8)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "Training and Prediction Time.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_misclassification_counts(rows):
    """比较各模型将 8 判为 9 与将 9 判为 8 的错误数量。"""
    names = [row["model"] for row in rows]
    eight_as_nine = [int(row["errors_8_as_9"]) for row in rows]
    nine_as_eight = [int(row["errors_9_as_8"]) for row in rows]
    x = np.arange(len(names))
    width = 0.36

    figure, axis = plt.subplots(figsize=(9, 5))
    bars_8_to_9 = axis.bar(x - width / 2, eight_as_nine, width, label="True 8 predicted as 9")
    bars_9_to_8 = axis.bar(x + width / 2, nine_as_eight, width, label="True 9 predicted as 8")
    axis.set_xticks(x, names)
    axis.set_ylabel("Number of test samples")
    axis.set_title("Directional Misclassification Counts")
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    axis.legend()
    axis.bar_label(bars_8_to_9, padding=3)
    axis.bar_label(bars_9_to_8, padding=3)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "Misclassification Counts.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 使用验证集绘制损失曲线；该过程不参与测试集评价。
    hinge_for_curve = HingeClassifier(**FINAL_PARAMETERS["Hinge Loss"])
    hinge_for_curve.fit(
        X_train,
        np.where(y_train == 9, 1, -1),
        epochs=EPOCHS,
        X_val=X_val,
        y_val=np.where(y_val == 9, 1, -1),
    )
    ce_for_curve = CE_Classifier(**FINAL_PARAMETERS["Cross-Entropy"])
    ce_for_curve.fit(
        X_train,
        np.where(y_train == 9, 1, 0),
        epochs=EPOCHS,
        X_val=X_val,
        y_val=np.where(y_val == 9, 1, 0),
    )
    plot_loss_curves(hinge_for_curve, ce_for_curve)

    # 最终训练使用合并后的训练集和验证集，测试集只在此处使用一次。
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    rows = []
    for model_name in MODEL_NAMES:
        start = perf_counter()
        model = train_final_model(model_name, X_full, y_full)
        train_seconds = perf_counter() - start

        start = perf_counter()
        prediction = predict_original_labels(model_name, model, X_test)
        prediction_seconds = perf_counter() - start
        errors_8_as_9 = int(np.sum((y_test == 8) & (prediction == 9)))
        errors_9_as_8 = int(np.sum((y_test == 9) & (prediction == 8)))

        row = {
            "model": model_name,
            "accuracy": f"{accuracy_score(y_test, prediction):.6f}",
            "precision": f"{precision_score(y_test, prediction, pos_label=9):.6f}",
            "recall": f"{recall_score(y_test, prediction, pos_label=9):.6f}",
            "f1": f"{f1_score(y_test, prediction, pos_label=9):.6f}",
            "training_seconds": f"{train_seconds:.6f}",
            "prediction_seconds": f"{prediction_seconds:.6f}",
            "errors_8_as_9": errors_8_as_9,
            "errors_9_as_8": errors_9_as_8,
            "total_errors": errors_8_as_9 + errors_9_as_8,
        }
        rows.append(row)
        print(
            f"{model_name}: F1={row['f1']}, training={row['training_seconds']} s, "
            f"prediction={row['prediction_seconds']} s, "
            f"8->9={errors_8_as_9}, 9->8={errors_9_as_8}"
        )
        plot_misclassified_samples(model_name, X_test, y_test, prediction)

    save_final_results(rows)
    plot_training_time(rows)
    plot_misclassification_counts(rows)


if __name__ == "__main__":
    main()

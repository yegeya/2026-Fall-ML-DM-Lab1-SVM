import os
import csv
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.hinge_model import HingeClassifier
from src.ce_model import CE_Classifier
from src.hyperparameter_search import search_ce_grid, search_hinge_grid
from src.svm_models import (
    search_linear_svm_grid,
    search_rbf_svm_grid,
    train_linear_svm,
    train_rbf_svm,
)


def save_search_results_csv(
    hinge_result,
    ce_result,
    linear_svm_result,
    rbf_svm_result,
    output_path="Hyperparameter Search Results.csv",
):
    """将全部参数组合及对应验证集 F1 保存为 CSV。"""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "learning_rate",
                "reg_lambda",
                "C",
                "gamma",
                "validation_f1",
            ]
        )

        for lr_index, learning_rate in enumerate(hinge_result["learning_rates"]):
            for reg_index, reg_lambda in enumerate(hinge_result["reg_lambdas"]):
                writer.writerow(
                    [
                        "Hinge Loss",
                        learning_rate,
                        reg_lambda,
                        "",
                        "",
                        hinge_result["f1_matrix"][lr_index, reg_index],
                    ]
                )

        for lr_index, learning_rate in enumerate(ce_result["learning_rates"]):
            for reg_index, reg_lambda in enumerate(ce_result["reg_lambdas"]):
                writer.writerow(
                    [
                        "Cross-Entropy",
                        learning_rate,
                        reg_lambda,
                        "",
                        "",
                        ce_result["f1_matrix"][lr_index, reg_index],
                    ]
                )

        for c_value, val_f1 in zip(
            linear_svm_result["c_values"],
            linear_svm_result["f1_scores"],
        ):
            writer.writerow(["Linear SVM", "", "", c_value, "", val_f1])

        for c_index, c_value in enumerate(rbf_svm_result["c_values"]):
            for gamma_index, gamma_value in enumerate(
                rbf_svm_result["gamma_values"]
            ):
                writer.writerow(
                    [
                        "RBF SVM",
                        "",
                        "",
                        c_value,
                        gamma_value,
                        rbf_svm_result["f1_matrix"][c_index, gamma_index],
                    ]
                )

X_train, X_val, X_test, y_train, y_val, y_test = load_data()

y_train_hinge = np.where(y_train == 9, 1, -1)
y_train_ce = np.where(y_train == 9, 1, 0)
y_val_hinge = np.where(y_val == 9, 1, -1)
y_val_ce = np.where(y_val == 9, 1, 0)

# 第二阶段：围绕第一阶段最优区域细化搜索。
HINGE_LEARNING_RATES = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
HINGE_REG_LAMBDAS = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.01]

# Cross-Entropy 的第一阶段最佳点位于搜索边界，向更大学习率和更小正则化扩展。
CE_LEARNING_RATES = [0.2, 0.3, 0.5, 0.7, 1.0]
CE_REG_LAMBDAS = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.01]

# SVM 的第一阶段最优点不在边界，保留原搜索范围。
SVM_C_VALUES = [0.01, 0.1, 1, 10, 100]
RBF_GAMMA_VALUES = ["scale", 0.001, 0.01, 0.1, 1]
GRID_SEARCH_EPOCHS = 1000

hinge_grid = search_hinge_grid(
    X_train,
    y_train_hinge,
    X_val,
    y_val_hinge,
    HINGE_LEARNING_RATES,
    HINGE_REG_LAMBDAS,
    epochs=GRID_SEARCH_EPOCHS,
)

ce_grid = search_ce_grid(
    X_train,
    y_train_ce,
    X_val,
    y_val_ce,
    CE_LEARNING_RATES,
    CE_REG_LAMBDAS,
    epochs=GRID_SEARCH_EPOCHS,
)

linear_svm_grid = search_linear_svm_grid(
    X_train,
    y_train,
    X_val,
    y_val,
    c_values=SVM_C_VALUES,
)

rbf_svm_grid = search_rbf_svm_grid(
    X_train,
    y_train,
    X_val,
    y_val,
    c_values=SVM_C_VALUES,
    gamma_values=RBF_GAMMA_VALUES,
)

save_search_results_csv(
    hinge_grid,
    ce_grid,
    linear_svm_grid,
    rbf_svm_grid,
    output_path="Refined Hyperparameter Search Results.csv",
)

print(
    "\nBest Hinge Loss parameters: "
    f"learning_rate={hinge_grid['best_learning_rate']}, "
    f"reg_lambda={hinge_grid['best_reg_lambda']}, "
    f"validation_f1={hinge_grid['best_f1']:.6f}"
)
print(
    "Best Cross-Entropy parameters: "
    f"learning_rate={ce_grid['best_learning_rate']}, "
    f"reg_lambda={ce_grid['best_reg_lambda']}, "
    f"validation_f1={ce_grid['best_f1']:.6f}"
)
print(
    "Best Linear SVM parameters: "
    f"C={linear_svm_grid['best_c']}, "
    f"validation_f1={linear_svm_grid['best_f1']:.6f}"
)
print(
    "Best RBF SVM parameters: "
    f"C={rbf_svm_grid['best_c']}, "
    f"gamma={rbf_svm_grid['best_gamma']}, "
    f"validation_f1={rbf_svm_grid['best_f1']:.6f}"
)

hinge_model = HingeClassifier(
    learning_rate=hinge_grid["best_learning_rate"],
    reg_lambda=hinge_grid["best_reg_lambda"],
)

ce_model = CE_Classifier(
    learning_rate=ce_grid["best_learning_rate"],
    reg_lambda=ce_grid["best_reg_lambda"],
)

hinge_model.fit(
    X_train,
    y_train_hinge,
    epochs=GRID_SEARCH_EPOCHS,
    X_val=X_val,
    y_val=y_val_hinge,
)
ce_model.fit(
    X_train,
    y_train_ce,
    epochs=GRID_SEARCH_EPOCHS,
    X_val=X_val,
    y_val=y_val_ce,
)

print(len(hinge_model.loss_history))
print(len(ce_model.loss_history))


# 获取两种模型的损失历史
hinge_loss = hinge_model.loss_history
hinge_val_loss = hinge_model.val_loss_history
ce_loss = ce_model.loss_history
ce_val_loss = ce_model.val_loss_history

epochs_hinge = np.arange(1, len(hinge_loss) + 1)
epochs_ce = np.arange(1, len(ce_loss) + 1)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(9, 5))

ax1.plot(epochs_hinge, hinge_loss, label="Hinge Loss")
ax1.plot(epochs_hinge, hinge_val_loss, label="Hinge Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Hinge Loss: Training vs Validation")

ax2.plot(epochs_ce, ce_loss, label="CE Loss")
ax2.plot(epochs_ce, ce_val_loss, label="CE Validation Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_title("CE Loss: Training vs Validation")

ax1.legend()
ax2.legend()

ax1.grid(True, alpha=0.3)
ax2.grid(True, alpha=0.3)

ax1.set_xlim(1, GRID_SEARCH_EPOCHS)
ax2.set_xlim(1, GRID_SEARCH_EPOCHS)

plt.tight_layout()

fig.savefig("Loss.png", dpi=300, bbox_inches="tight")
plt.show()


def plot_grid_heatmap(axis, grid_result, title, vmin, vmax):
    """在指定子图中绘制学习率和正则化系数的验证集 F1 热力图。"""
    f1_matrix = grid_result["f1_matrix"]
    image = axis.imshow(
        f1_matrix,
        cmap="YlGnBu",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    learning_rates = grid_result["learning_rates"]
    reg_lambdas = grid_result["reg_lambdas"]
    axis.set_xticks(np.arange(len(reg_lambdas)), labels=[f"{x:g}" for x in reg_lambdas])
    axis.set_yticks(
        np.arange(len(learning_rates)),
        labels=[f"{x:g}" for x in learning_rates],
    )
    axis.set_xlabel("Regularization Coefficient (lambda)")
    axis.set_ylabel("Learning Rate")
    axis.set_title(title)

    for row_index in range(f1_matrix.shape[0]):
        for column_index in range(f1_matrix.shape[1]):
            value = f1_matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.4f}",
                ha="center",
                va="center",
                color="white" if value < f1_matrix.mean() else "black",
                fontsize=8,
            )

    best_row = learning_rates.index(grid_result["best_learning_rate"])
    best_column = reg_lambdas.index(grid_result["best_reg_lambda"])
    axis.add_patch(
        Rectangle(
            (best_column - 0.5, best_row - 0.5),
            1,
            1,
            fill=False,
            edgecolor="crimson",
            linewidth=3,
        )
    )

    return image


fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
manual_f1_min = min(hinge_grid["f1_matrix"].min(), ce_grid["f1_matrix"].min())
manual_f1_max = max(hinge_grid["f1_matrix"].max(), ce_grid["f1_matrix"].max())
image1 = plot_grid_heatmap(
    ax1,
    hinge_grid,
    "Hinge Loss Validation F1",
    manual_f1_min,
    manual_f1_max,
)
image2 = plot_grid_heatmap(
    ax2,
    ce_grid,
    "Cross-Entropy Validation F1",
    manual_f1_min,
    manual_f1_max,
)

fig2.colorbar(image1, ax=ax1, label="Validation F1-score")
fig2.colorbar(image2, ax=ax2, label="Validation F1-score")
fig2.tight_layout()
fig2.savefig(
    "F1 Score Refined Grid Search.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(
    linear_svm_grid["c_values"],
    linear_svm_grid["f1_scores"],
    marker="o",
)
ax1.scatter(
    linear_svm_grid["best_c"],
    linear_svm_grid["best_f1"],
    color="crimson",
    marker="*",
    s=180,
    zorder=3,
    label=(
        f"Best C={linear_svm_grid['best_c']:g}, "
        f"F1={linear_svm_grid['best_f1']:.4f}"
    ),
)
ax1.set_xscale("log")
ax1.set_xlabel("C")
ax1.set_ylabel("Validation F1-score")
ax1.set_title("Linear SVM Parameter Search")
ax1.grid(True, alpha=0.3)
ax1.legend()

rbf_f1_matrix = rbf_svm_grid["f1_matrix"]
image = ax2.imshow(rbf_f1_matrix, cmap="YlGnBu", aspect="auto")
ax2.set_xticks(
    np.arange(len(rbf_svm_grid["gamma_values"])),
    labels=[str(value) for value in rbf_svm_grid["gamma_values"]],
)
ax2.set_yticks(
    np.arange(len(rbf_svm_grid["c_values"])),
    labels=[f"{value:g}" for value in rbf_svm_grid["c_values"]],
)
ax2.set_xlabel("gamma")
ax2.set_ylabel("C")
ax2.set_title("RBF SVM Validation F1")

for row_index in range(rbf_f1_matrix.shape[0]):
    for column_index in range(rbf_f1_matrix.shape[1]):
        value = rbf_f1_matrix[row_index, column_index]
        ax2.text(
            column_index,
            row_index,
            f"{value:.4f}",
            ha="center",
            va="center",
            color="white" if value < rbf_f1_matrix.mean() else "black",
            fontsize=8,
        )

best_c_row = rbf_svm_grid["c_values"].index(rbf_svm_grid["best_c"])
best_gamma_column = rbf_svm_grid["gamma_values"].index(
    rbf_svm_grid["best_gamma"]
)
ax2.add_patch(
    Rectangle(
        (best_gamma_column - 0.5, best_c_row - 0.5),
        1,
        1,
        fill=False,
        edgecolor="crimson",
        linewidth=3,
    )
)
fig3.colorbar(image, ax=ax2, label="Validation F1-score")
fig3.tight_layout()
fig3.savefig("SVM Parameter Search.png", dpi=300, bbox_inches="tight")
plt.show()


X_full = np.concatenate([X_train, X_val], axis=0)
y_full = np.concatenate([y_train, y_val], axis=0)

final_linear = train_linear_svm(
    X_full,
    y_full,
    c_value=linear_svm_grid["best_c"],
)
final_rbf = train_rbf_svm(
    X_full,
    y_full,
    c_value=rbf_svm_grid["best_c"],
    gamma=rbf_svm_grid["best_gamma"],
)

final_hinge = HingeClassifier(
    learning_rate=hinge_grid["best_learning_rate"],
    reg_lambda=hinge_grid["best_reg_lambda"],
)
final_hinge.fit(
    X_full,
    np.where(y_full == 9, 1, -1),
    epochs=GRID_SEARCH_EPOCHS,
)

final_ce = CE_Classifier(
    learning_rate=ce_grid["best_learning_rate"],
    reg_lambda=ce_grid["best_reg_lambda"],
)
final_ce.fit(
    X_full,
    np.where(y_full == 9, 1, 0),
    epochs=GRID_SEARCH_EPOCHS,
)

y_pred_linear = final_linear.predict(X_test)
y_pred_rbf = final_rbf.predict(X_test)
y_pred_hinge = np.where(final_hinge.predict(X_test) == 1, 9, 8)
y_pred_ce = np.where(final_ce.predict(X_test) == 1, 9, 8)

models = ["Linear SVM", "RBF SVM", "Hinge Loss", "CE Loss"]
predictions = [y_pred_linear, y_pred_rbf, y_pred_hinge, y_pred_ce]
cms = [
    confusion_matrix(y_test, prediction, labels=[8, 9])
    for prediction in predictions
]
titles = ["Linear SVM", "RBF SVM", "Hinge Loss", "Cross-entropy"]

accuracy = [accuracy_score(y_test, prediction) for prediction in predictions]
precision = [
    precision_score(y_test, prediction, pos_label=9)
    for prediction in predictions
]
recall = [
    recall_score(y_test, prediction, pos_label=9)
    for prediction in predictions
]
f1 = [
    f1_score(y_test, prediction, pos_label=9)
    for prediction in predictions
]

print("\nFinal test-set results")
for name, acc, pre, rec, f1_value in zip(
    models,
    accuracy,
    precision,
    recall,
    f1,
):
    print(
        f"{name}: accuracy={acc:.6f}, precision={pre:.6f}, "
        f"recall={rec:.6f}, f1={f1_value:.6f}"
    )

fig4, axes = plt.subplots(2, 2, figsize=(9, 8))

for ax, cm, title in zip(axes.flat, cms, titles):

    # 1. 显示混淆矩阵
    ax.imshow(cm, cmap="Blues")

    # 2. 设置标题
    ax.set_title(title)

    # 3. 设置坐标轴
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1], labels=["8", "9"])
    ax.set_yticks([0, 1], labels=["8", "9"])

    # 4. 在每个格子里显示数字
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12
            )

plt.tight_layout()

fig4.savefig(
    "Confusion_Matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


x = np.arange(len(models))
width = 0.2

fig5, ax = plt.subplots(figsize=(11, 5))

ax.bar(x - 1.5 * width, accuracy, width, label="Accuracy")
ax.bar(x - 0.5 * width, precision, width, label="Precision")
ax.bar(x + 0.5 * width, recall, width, label="Recall")
ax.bar(x + 1.5 * width, f1, width, label="F1")

ax.set_ylabel("Score")
ax.set_title("Performance Comparison of Four Models")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
ax.set_axisbelow(True)

fig5.show()
fig5.savefig("main.png", dpi=300, bbox_inches="tight")
# 显示四个模型的名称
ax.set_xticks(x)
ax.set_xticklabels(models)

# 放大纵坐标范围
ax.set_ylim(0.95, 1.0)

# 设置标题和坐标轴
ax.set_title("Performance Comparison of Four Models")
ax.set_ylabel("Score")

# 把图例放到图表上方
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4
)

# 只显示水平网格
ax.grid(True, axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()

fig6, ax = plt.subplots(figsize=(10, 5))

ax.plot(models, accuracy, marker="o", label="Accuracy")
ax.plot(models, precision, marker="s", label="Precision")
ax.plot(models, recall, marker="^", label="Recall")
ax.plot(models, f1, marker="D", label="F1-score")

ax.set_ylim(0.97, 1.0)
ax.set_ylabel("Score")
ax.set_title("Performance Comparison of Four Models")

ax.legend(ncol=4, loc="lower center",
          bbox_to_anchor=(0.5, 1.02))

ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
fig6.savefig("Model_Performance_Line.png",
             dpi=300, bbox_inches="tight")
plt.show()

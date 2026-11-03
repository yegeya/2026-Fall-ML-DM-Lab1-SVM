"""
二维网格搜索工具：在验证集上联合选择学习率和 L2 正则化系数。
"""

import numpy as np
from sklearn.metrics import f1_score

from ce_model import CE_Classifier
from hinge_model import HingeClassifier


def _run_grid_search(model_class, X_train, y_train, X_val, y_val,
                     learning_rates, reg_lambdas, epochs, model_name):
    """在所有 (learning_rate, reg_lambda) 组合上训练并返回验证集 F1 矩阵。"""
    f1_matrix = np.zeros(
        (len(learning_rates), len(reg_lambdas)),
        dtype=np.float64,
    )
    best_f1 = -np.inf
    best_learning_rate = None
    best_reg_lambda = None

    print(f"\nBegin {model_name} grid search")
    for lr_index, learning_rate in enumerate(learning_rates):
        for reg_index, reg_lambda in enumerate(reg_lambdas):
            model = model_class(
                learning_rate=learning_rate,
                reg_lambda=reg_lambda,
            )
            model.fit(X_train, y_train, epochs=epochs)

            y_val_pred = model.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred, pos_label=1)
            f1_matrix[lr_index, reg_index] = val_f1

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_learning_rate = learning_rate
                best_reg_lambda = reg_lambda

            print(
                f"learning_rate={learning_rate:<7g} "
                f"reg_lambda={reg_lambda:<7g} "
                f"validation_f1={val_f1:.6f}"
            )

    return {
        "learning_rates": list(learning_rates),
        "reg_lambdas": list(reg_lambdas),
        "f1_matrix": f1_matrix,
        "best_learning_rate": best_learning_rate,
        "best_reg_lambda": best_reg_lambda,
        "best_f1": best_f1,
    }


def search_hinge_grid(X_train, y_train, X_val, y_val,
                      learning_rates, reg_lambdas, epochs=1000):
    """联合搜索 Hinge Loss 模型的学习率和正则化系数。"""
    return _run_grid_search(
        HingeClassifier,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rates,
        reg_lambdas,
        epochs,
        "Hinge Loss",
    )


def search_ce_grid(X_train, y_train, X_val, y_val,
                   learning_rates, reg_lambdas, epochs=1000):
    """联合搜索 Cross-Entropy 模型的学习率和正则化系数。"""
    return _run_grid_search(
        CE_Classifier,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rates,
        reg_lambdas,
        epochs,
        "Cross-Entropy",
    )

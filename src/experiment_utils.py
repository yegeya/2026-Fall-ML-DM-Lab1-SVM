"""最终模型训练与评价流程共用的配置和辅助函数。"""

import numpy as np

from .ce_model import CE_Classifier
from .hinge_model import HingeClassifier
from .svm_models import train_linear_svm, train_rbf_svm


EPOCHS = 1000
MODEL_NAMES = ("Linear SVM", "RBF SVM", "Hinge Loss", "Cross-Entropy")

# 参数由训练集内部的验证集搜索确定；测试集仅用于最终评价。
FINAL_PARAMETERS = {
    "Linear SVM": {"C": 0.1},
    "RBF SVM": {"C": 10, "gamma": "scale"},
    "Hinge Loss": {"learning_rate": 0.4, "reg_lambda": 0.00003},
    "Cross-Entropy": {"learning_rate": 1.5, "reg_lambda": 0.00001},
}


def train_final_model(model_name, X_train, y_train, X_val=None, y_val=None):
    """按固定的最终参数训练指定模型。"""
    if model_name == "Linear SVM":
        return train_linear_svm(X_train, y_train, FINAL_PARAMETERS[model_name]["C"])
    if model_name == "RBF SVM":
        parameters = FINAL_PARAMETERS[model_name]
        return train_rbf_svm(X_train, y_train, parameters["C"], parameters["gamma"])
    if model_name == "Hinge Loss":
        fit_options = {}
        if X_val is not None and y_val is not None:
            fit_options = {"X_val": X_val, "y_val": np.where(y_val == 9, 1, -1)}
        return HingeClassifier(**FINAL_PARAMETERS[model_name]).fit(
            X_train, np.where(y_train == 9, 1, -1), epochs=EPOCHS, **fit_options
        )
    if model_name == "Cross-Entropy":
        fit_options = {}
        if X_val is not None and y_val is not None:
            fit_options = {"X_val": X_val, "y_val": np.where(y_val == 9, 1, 0)}
        return CE_Classifier(**FINAL_PARAMETERS[model_name]).fit(
            X_train, np.where(y_train == 9, 1, 0), epochs=EPOCHS, **fit_options
        )
    raise ValueError(f"不支持的模型名称：{model_name}")


def train_all_final_models(X_train, y_train):
    """训练全部四个固定参数模型，供重复验证等无需单模型计时的流程使用。"""
    return {name: train_final_model(name, X_train, y_train) for name in MODEL_NAMES}


def restore_original_labels(model_name, prediction):
    """将各模型的内部预测标签统一还原为原始数字标签 8、9。"""
    if "SVM" in model_name:
        return prediction
    return np.where(prediction == 1, 9, 8)


def predict_original_labels(model_name, model, X):
    """执行预测并将输出统一还原为原始数字标签 8、9。"""
    return restore_original_labels(model_name, model.predict(X))

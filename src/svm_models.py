# 此程序用以实现线性核与RBF核
# 此程序中可以直接调用sklearn的包
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from .data_loader import load_data
import numpy as np

def train_linear_svm(X_train, y_train, c_value):
    model = SVC(kernel="linear", C=c_value)     # 参数待调整
    model.fit(X_train, y_train)
    return model

def train_rbf_svm(X_train, y_train, c_value, gamma):
    model = SVC(kernel="rbf", C=c_value, gamma=gamma)     # 参数待调整
    model.fit(X_train, y_train)
    return model

# 改进方向：这里虽然有一万多的样本，但是不确定这个样本数算多还是少
# 如果是比较少的话，可以使用K折交叉验证
# 或者可以使用AIC、BIC信息量准则（不过这个好像只能用于概率模型？）

def search_linear_svm(X_train, y_train, X_val, y_val):
    C_values = [0.01, 0.1, 1, 10, 100]
    best_C = None
    best_f1 = -1

    print("begin search linear svm")
    for C in C_values:
        ln_svm = train_linear_svm(X_train, y_train, C)
        ln_svm_y_pred = ln_svm.predict(X_val)
        # ln_svm_acc = accuracy_score(y_val, ln_svm_y_pred)
        ln_svm_f1 = f1_score(y_val, ln_svm_y_pred, pos_label=9)
        if ln_svm_f1 > best_f1:
            best_C = C
            best_f1 = ln_svm_f1

        print(f"C = {C}    f1 = {ln_svm_f1:.4f}")

    return best_C, best_f1

def search_rbf_svm(X_train, y_train, X_val, y_val):
    C_values = [0.01, 0.1, 1, 10, 100]
    gamma_values = ["scale", 0.001, 0.01, 0.1, 1]

    best_C = None
    best_gamma = None
    best_f1 = -1

    print("begin search rbf svm")
    for C in C_values:
        for gamma in gamma_values:
            rbf_svm = train_rbf_svm(X_train, y_train, C, gamma)
            rbf_svm_y_pred = rbf_svm.predict(X_val)
            rbf_svm_f1 = f1_score(y_val, rbf_svm_y_pred, pos_label=9)
            if rbf_svm_f1 > best_f1:
                best_f1 = rbf_svm_f1
                best_C = C
                best_gamma = gamma

            print(f"C = {C}    gamma = {gamma}    f1 = {rbf_svm_f1:.4f}")

    return best_C, best_gamma, best_f1


def search_linear_svm_grid(X_train, y_train, X_val, y_val,
                           c_values=(0.01, 0.1, 1, 10, 100)):
    """搜索 Linear SVM 的 C，并返回完整验证集 F1 结果。"""
    f1_scores = np.zeros(len(c_values), dtype=np.float64)
    best_f1 = -np.inf
    best_c = None

    print("\nBegin Linear SVM parameter search")
    for index, c_value in enumerate(c_values):
        model = train_linear_svm(X_train, y_train, c_value)
        y_val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, pos_label=9)
        f1_scores[index] = val_f1

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_c = c_value

        print(f"C={c_value:<7g} validation_f1={val_f1:.6f}")

    return {
        "c_values": list(c_values),
        "f1_scores": f1_scores,
        "best_c": best_c,
        "best_f1": best_f1,
    }


def search_rbf_svm_grid(X_train, y_train, X_val, y_val,
                        c_values=(0.01, 0.1, 1, 10, 100),
                        gamma_values=("scale", 0.001, 0.01, 0.1, 1)):
    """联合搜索 RBF SVM 的 C 和 gamma，并返回完整验证集 F1 矩阵。"""
    f1_matrix = np.zeros(
        (len(c_values), len(gamma_values)),
        dtype=np.float64,
    )
    best_f1 = -np.inf
    best_c = None
    best_gamma = None

    print("\nBegin RBF SVM grid search")
    for c_index, c_value in enumerate(c_values):
        for gamma_index, gamma_value in enumerate(gamma_values):
            model = train_rbf_svm(X_train, y_train, c_value, gamma_value)
            y_val_pred = model.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred, pos_label=9)
            f1_matrix[c_index, gamma_index] = val_f1

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_c = c_value
                best_gamma = gamma_value

            print(
                f"C={c_value:<7g} gamma={str(gamma_value):<7} "
                f"validation_f1={val_f1:.6f}"
            )

    return {
        "c_values": list(c_values),
        "gamma_values": list(gamma_values),
        "f1_matrix": f1_matrix,
        "best_c": best_c,
        "best_gamma": best_gamma,
        "best_f1": best_f1,
    }


def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}")
    print("Acurracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, pos_label=9))
    print("Recall:", recall_score(
        y_true, y_pred, pos_label=9
    ))
    print("F1-score:", f1_score(
        y_true, y_pred, pos_label=9
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(
        y_true, y_pred, labels=[8, 9]
    ))      

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # best_ln_C, best_ln_f1 = search_linear_svm(X_train, y_train, X_val, y_val)
    # best_rbf_C, best_rbf_gamma, best_rbf_f1 = search_rbf_svm(X_train, y_train, X_val, y_val)

    # print(f"best_ln_C: {best_ln_C}, best_ln_f1: {best_ln_f1}\n")
    # print(f"best_rbf_C: {best_rbf_C}, best_rbf_gamma: {best_rbf_gamma}, best_rbf_f1: {best_rbf_f1}")

    X_full_train = np.concatenate([X_train, X_val], axis=0)
    y_full_train = np.concatenate([y_train, y_val], axis=0)

    # 此处使用的超参数通过训练集内部的验证集搜索得到
    final_linear = train_linear_svm(X_full_train, y_full_train, 0.1)
    final_rbf = train_rbf_svm(X_full_train, y_full_train, 10, "scale")

    final_ln_y_pred = final_linear.predict(X_test)
    final_rbf_y_pred = final_rbf.predict(X_test)

    evaluate_model("linear_svm", y_test, final_ln_y_pred)
    evaluate_model("rbf_svm",y_test, final_rbf_y_pred)

    # linear_svm = train_linear_svm(X_train, y_train)
    # rbf_svm = train_rbf_svm(X_train, y_train)

    # linear_svm_y_pred = linear_svm.predict(X_val)
    # linear_svm_accuracy = accuracy_score(y_val, linear_svm_y_pred)
    # linear_svm_f1 = f1_score(y_val, linear_svm_y_pred, pos_label=9)

    # rbf_svm_y_pred = rbf_svm.predict(X_val)
    # rbf_svm_accuracy = accuracy_score(y_val, rbf_svm_y_pred)
    # rbf_svm_f1 = f1_score(y_val, rbf_svm_y_pred, pos_label=9)

    # print(f"linear_svm:\naccuracy: {linear_svm_accuracy}\nf1_score: {linear_svm_f1}\n")
    # print(f"rbf_svm:\naccuracy: {rbf_svm_accuracy}\nf1_score: {rbf_svm_f1}\n")


if __name__ == "__main__":
    main()


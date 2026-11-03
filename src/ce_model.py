# 此程序用以手动实现 Cross-entropy 与梯度下降
import numpy as np
from .data_loader import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class CE_Classifier:
    def __init__(self, learning_rate=0.01, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.w = None
        self.b = 0.0

    def initialize(self, n_features):
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

    def decision_function(self, X):
        return X @ self.w + self.b

    def compute_sigmoid(self, z):
        return 1/(1+ np.exp(-z))

    def compute_CE_loss(self, X, y):
        z = self.decision_function(X)
        p =  self.compute_sigmoid(z)
        # p 为 float32 时，1 - 1e-12 会被舍入为 1.0。
        # 使用该数据类型的机器精度，避免计算 log(1 - p) 时出现 log(0)。
        eps = np.finfo(p.dtype).eps
        p = np.clip(p, eps, 1 - eps)
        loss = -(y*np.log(p)+(1-y)*np.log(1-p))
        avg_loss = np.mean(loss)
        reg = 0.5*self.reg_lambda*np.sum(self.w**2)
        return avg_loss + reg

    def compute_gradient(self, X, y):
        z = self.decision_function(X)
        p = self.compute_sigmoid(z)
        error = p - y
        grad_w = X.T @ error/len(X) + self.reg_lambda*self.w
        grad_b = np.mean(error)

        return grad_w, grad_b

    
    def fit(self, X, y, epochs=100, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.initialize(X.shape[1])

        self.loss_history = []
        self.val_loss_history = []
        self.val_f1_history = []
        self.val_accuracy_history = []

        # 检查训练标签
        assert np.all(np.isin(y, [0, 1])), \
            "Training labels must be 0 or 1"

        has_val = X_val is not None and y_val is not None

        if has_val:
            assert np.all(np.isin(y_val, [0, 1])), \
                "Validation labels must be 0 or 1"

        for epoch in range(epochs):

            # 1. 计算梯度
            grad_w, grad_b = self.compute_gradient(X, y)

            # 2. 更新参数
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

            # 3. 记录训练损失
            train_loss = self.compute_CE_loss(X, y)
            self.loss_history.append(train_loss)

            if has_val:
                val_loss = self.compute_CE_loss(X_val, y_val)
                self.val_loss_history.append(val_loss)

            # 4. 每100轮记录一次验证集表现
            # if (epoch + 1) % 100 == 0:

            #     print(f"\nEpoch {epoch + 1}/{epochs}")
            #     print(f"Train loss: {train_loss:.6f}")

            #     if has_val:
            #         val_loss = self.compute_CE_loss(X_val, y_val)
            #         y_val_pred = self.predict(X_val)

            #         val_acc = accuracy_score(y_val, y_val_pred)
            #         val_f1 = f1_score(
            #             y_val, y_val_pred, pos_label=1
            #         )

            #         self.val_loss_history.append(val_loss)
            #         self.val_accuracy_history.append(val_acc)
            #         self.val_f1_history.append(val_f1)

            #         print(f"Validation loss: {val_loss:.6f}")
            #         print(f"Validation accuracy: {val_acc:.6f}")
            #         print(f"Validation F1: {val_f1:.6f}")

        return self

    def predict(self, X):
        z = self.decision_function(X)
        p = self.compute_sigmoid(z)
        return np.where(p >= 0.5, 1, 0)

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # y_train_ce = np.where(y_train == 9, 1, 0)
    # y_val_ce = np.where(y_val == 9, 1, 0)

    # model = CE_Classifier()
    # model.initialize(X_train.shape[1])
    # print("Unique labels:", np.unique(y_train_ce))
    # print("Initial CE loss:", model.compute_CE_loss(X_train, y_train_ce))
    # model.fit(X_train, y_train_ce, 1000, X_val, y_val_ce)

    # y_val_pred = model.predict(X_val)

    # print("\nValidation Results:")
    # print("Accuracy:", accuracy_score(y_val_ce, y_val_pred))
    # print("Precision:", precision_score(y_val_ce, y_val_pred, pos_label=1))
    # print("Recall:", recall_score(y_val_ce, y_val_pred, pos_label=1))
    # print("F1-score:", f1_score(y_val_ce, y_val_pred, pos_label=1))

    # print("Confusion Matrix:")
    # print(confusion_matrix(y_val_ce, y_val_pred, labels=[0, 1]))

    # print("Validation loss:", model.compute_CE_loss(X_val, y_val_ce))
    # learning_rates = [0.2, 0.3, 0.5]

    # for lr in learning_rates:
    #     model = CE_Classifier(
    #         learning_rate=lr,
    #         reg_lambda=0.001
    #     )

    #     model.fit(
    #         X_train,
    #         y_train_ce,
    #         epochs=1000,
    #         X_val=X_val,
    #         y_val=y_val_ce
    #     )

    #     y_pred = model.predict(X_val)
    #     f1 = f1_score(y_val_ce, y_pred, pos_label=1)

    #     print(f"Learning rate: {lr}, Validation F1: {f1:.6f}")
    # reg_lambdas = [0.0001, 0.001, 0.01, 0.1]

    # for reg in reg_lambdas:
    #     model = CE_Classifier(
    #         learning_rate=0.1,
    #         reg_lambda=reg
    #     )
    #     model.fit(X_train, y_train_ce, epochs=1000)
    #     y_pred = model.predict(X_val)
    #     f1 = f1_score(y_val_ce, y_pred, pos_label=1)
    #     print(f"Lambda: {reg}, Validation F1: {f1:.6f}")

    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    print(X_full.shape)  # (11800, 784)

    y_full_ce = np.where(y_full == 9, 1, 0)

    ce_model = CE_Classifier(
        learning_rate=0.5,
        reg_lambda=0.001
    )

    ce_model.fit(
        X_full,
        y_full_ce,
        epochs=1000
    )

    y_test_ce = np.where(y_test == 9, 1, 0)
    y_pred = ce_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test_ce, y_pred))
    print("Precision:", precision_score(y_test_ce, y_pred, pos_label=1))
    print("Recall:", recall_score(y_test_ce, y_pred, pos_label=1))
    print("F1:", f1_score(y_test_ce, y_pred, pos_label=1))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_ce, y_pred, labels=[0, 1]))

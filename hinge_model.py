# 此程序用以手动实现 Hinge loss 与梯度下降
import numpy as np
from data_loader import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class HingeClassifier:

    def __init__(self, learning_rate = 0.01, reg_lambda = 0.01):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.w = None
        self.b = 0.0

    def initialize(self, n_feature):
        self.w = np.zeros(n_feature, dtype=np.float32)
        self.b = 0.0

    def decision_function(self, X):
        return X @ self.w + self.b

    def compute_loss(self, X, y):
        margin = y * self.decision_function(X)

        avg = np.mean(np.maximum(0, 1 - margin))

        reg = 0.5 * self.reg_lambda * np.sum(self.w ** 2)

        return avg + reg

    def compute_gradient(self, X, y):
        margin = y * self.decision_function(X)

        mask = margin < 1

        X_active = X[mask]
        y_active = y[mask]

        grad_w = -1/len(X)*(X_active.T @ y_active) + self.reg_lambda * self.w
        grad_b = -1/len(X)*np.sum(y_active)

        return grad_w, grad_b

    def fit(self, X, y, epochs = 100, X_val = None, y_val = None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.initialize(X.shape[1])

        self.loss_history = []
        self.val_loss_history = []
        self.val_f1_history = []
        self.val_accuracy_history = []

        assert np.all(np.isin(y, [-1, 1])), \
            "Training labels must be 0 or 1"
        
        has_val = X_val is not None and y_val is not None
        
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)

            assert np.all(np.isin(y_val, [-1, 1])), \
                "Validation labels must be -1 or 1"

        for epoch in range(epochs):
            # 1. 计算训练集梯度
            grad_w, grad_b = self.compute_gradient(X, y)

            # 2. 更新模型参数
            self.w = self.w - self.learning_rate * grad_w
            self.b = self.b - self.learning_rate * grad_b

            # 3. 记录训练损失
            train_loss = self.compute_loss(X, y)
            self.loss_history.append(train_loss)

            # 4. 记录验证损失
            if has_val:
                val_loss = self.compute_loss(X_val, y_val)
                self.val_loss_history.append(val_loss)

            # if (epoch + 1) % 100 == 0:
            #     print(f"Epoch{epoch + 1}/{epochs},"
            #           f"Train Loss: {self.loss_history[-1]:.6f}")
            #     if has_val:
            #         val_loss = self.compute_loss(X_val, y_val)
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
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)
    

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # y_train_hinge = np.where(y_train == 9, 1, -1)
    # y_val_hinge = np.where(y_val == 9, 1, -1)

#     model = HingeClassifier(learning_rate=0.01, reg_lambda=0.01)
#     model.fit(X_train, y_train_hinge, 1000, X_val, y_val_hinge)

#     print("Inittial loss:", 1.0)
#     print("Final loss:", model.compute_loss(X_train, y_train_hinge))

    

#     y_val_pred = model.predict(X_val)

#     print("\nValidation Results:")
#     print("Accuracy:", accuracy_score(y_val_hinge, y_val_pred))
#     print("Precision:", precision_score(
#         y_val_hinge, y_val_pred, pos_label=1
#     ))
#     print("Recall:", recall_score(
#         y_val_hinge, y_val_pred, pos_label=1
#     ))
#     print("F1-score:", f1_score(
#         y_val_hinge, y_val_pred, pos_label=1
#     ))

#     print("Validation loss:", model.compute_loss(
#         X_val, y_val_hinge
#     ))

#     print("Confusion Matrix:")
# print(confusion_matrix(y_val_hinge, y_val_pred, labels=[-1, 1]))
    # learning_rates = [0.2, 0.3, 0.5]

    # for lr in learning_rates:
    #     model = HingeClassifier(
    #         learning_rate=lr,
    #         reg_lambda=0.01
    #     )

    #     model.fit(
    #         X_train,
    #         y_train_hinge,
    #         epochs=1000,
    #         X_val=X_val,
    #         y_val=y_val_hinge
    #     )

    #     y_pred = model.predict(X_val)
    #     f1 = f1_score(y_val_hinge, y_pred, pos_label=1)

    #     print(f"Learning rate: {lr}, Validation F1: {f1:.6f}")
    # reg_lambdas = [0.0001, 0.001, 0.01, 0.1]

    # for reg in reg_lambdas:
    #     model = HingeClassifier(
    #         learning_rate=0.1,
    #         reg_lambda=reg
    #     )
    #     model.fit(X_train, y_train_hinge, epochs=1000)
    #     y_pred = model.predict(X_val)
    #     f1 = f1_score(y_val_hinge, y_pred, pos_label=1)
    #     print(f"Lambda: {reg}, Validation F1: {f1:.6f}")

    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    print(X_full.shape)  # (11800, 784)

    y_full_hinge = np.where(y_full == 9, 1, -1)

    hinge_model = HingeClassifier(
        learning_rate=0.2,
        reg_lambda=0.01
    )

    hinge_model.fit(
        X_full,
        y_full_hinge,
        epochs=1000
    )

    y_test_hinge = np.where(y_test == 9, 1, -1)
    y_pred = hinge_model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test_hinge, y_pred))
    print("Precision:", precision_score(y_test_hinge, y_pred, pos_label=1))
    print("Recall:", recall_score(y_test_hinge, y_pred, pos_label=1))
    print("F1:", f1_score(y_test_hinge, y_pred, pos_label=1))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_hinge, y_pred, labels=[-1, 1]))
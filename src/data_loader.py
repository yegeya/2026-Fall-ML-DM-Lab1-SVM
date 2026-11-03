# 此程序用以实现加载数据、归一化、划分验证集
# 首先把原始图像转化为二维矩阵，同时将像素归一化到[0,1]
from sklearn.model_selection import train_test_split
import torch
import numpy as np 
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed_8_9"


def load_training_data():
    """加载全部原始训练样本，供重复验证等不应接触测试集的流程使用。"""
    train_data = torch.load(
        DATA_DIR / "training.pt",
        map_location="cpu",
        weights_only=True,
    )
    X_train, y_train = train_data
    X_train = X_train.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = y_train.numpy()
    return X_train, y_train

def load_data():
    train_data = torch.load(DATA_DIR / "training.pt", map_location="cpu",weights_only=True)
    test_data = torch.load(DATA_DIR / "test.pt",map_location="cpu", weights_only=True)

    X_train, y_train = train_data
    X_test, y_test = test_data
    y_train = y_train.numpy()
    y_test = y_test.numpy()

    # 将tensor转换成numpy数组
    X_train = X_train.numpy()
    X_test = X_test.numpy()

    # 将图像展平成784维向量
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    # 将像素归一化到[0,1]
    X_train = X_train.astype(np.float32)/255.0
    X_test = X_test.astype(np.float32)/255.0 

    # 将原训练集划分为训练集和验证集
    X_train, X_val, y_train, y_val= train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=23,
        shuffle=True,
        stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    data = load_data()

    for name, arr in zip(
        ["X_train", "X_val", "X_test",
         "y_train", "y_val", "y_test"],
         data
    ):
        print(name, arr.shape)

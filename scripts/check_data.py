
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed_8_9"


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path.resolve()}")

    # 数据应为由图像 Tensor 和标签 Tensor 组成的元组或列表。
    # 仅加载可信来源的老师提供文件。
    try:
        data = torch.load(
            path, map_location="cpu", weights_only=True
        )
    except TypeError:
        # 兼容较旧版本的 PyTorch
        data = torch.load(path, map_location="cpu")

    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise ValueError("数据格式不符合预期，应为 (images, labels)")

    images, labels = data
    return images, labels


def inspect_dataset(name, images, labels):
    print(f"\n===== {name} =====")
    print("图像形状：", tuple(images.shape))
    print("标签形状：", tuple(labels.shape))
    print("图像类型：", images.dtype)
    print("标签类型：", labels.dtype)

    print("像素最小值：", images.min().item())
    print("像素最大值：", images.max().item())

    print("标签类别：", torch.unique(labels).tolist())

    values, counts = torch.unique(labels, return_counts=True)
    print("类别计数：", {
        int(v): int(c)
        for v, c in zip(values.tolist(), counts.tolist())
    })

    assert images.ndim == 3, "图像应为 (N, 28, 28)"
    assert images.shape[1:] == (28, 28), "图像尺寸不符合预期"
    assert labels.ndim == 1, "标签应为一维数组"
    assert len(images) == len(labels), "图像与标签数量不一致"
    assert set(torch.unique(labels).tolist()) == {8, 9}
    assert images.dtype == torch.uint8
    assert labels.dtype == torch.int64


def main():
    train_images, train_labels = load_dataset(
        DATA_DIR / "training.pt"
    )
    test_images, test_labels = load_dataset(
        DATA_DIR / "test.pt"
    )

    inspect_dataset("训练集", train_images, train_labels)
    inspect_dataset("测试集", test_images, test_labels)

    # 展示部分样本，人工检查图像和标签是否对应。
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for i, ax in enumerate(axes.flat):
        ax.imshow(train_images[i].numpy(), cmap="gray")
        ax.set_title(f"Label: {train_labels[i].item()}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

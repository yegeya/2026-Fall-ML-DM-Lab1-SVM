"""基于已有最终结果 CSV 重绘耗时图和错分数量图，不重新训练模型。"""

import csv

from final_evaluation import plot_misclassification_counts, plot_training_time


def main():
    with open("Final Test Results and Timing.csv", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Final Test Results and Timing.csv 中没有可绘制的数据。")

    plot_training_time(rows)
    plot_misclassification_counts(rows)


if __name__ == "__main__":
    main()

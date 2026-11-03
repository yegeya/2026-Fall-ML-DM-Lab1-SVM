# 机器学习与数据挖掘实验一：SVM 二分类实验

本项目以手写数字 8 与 9 为对象，比较 Linear SVM、RBF SVM、手写 Hinge Loss 分类器和手写 Cross-Entropy 分类器的分类表现。实验包含数据预处理、验证集参数搜索、最终测试、损失曲线、错分样本分析和多随机种子重复验证。

## 目录说明

- `src/`：数据读取、预处理、SVM 与两个手写分类器的核心实现；其中 `experiment_utils.py` 统一维护最终参数、训练轮数、模型训练和标签还原逻辑。
- `scripts/`：可直接执行的实验脚本。
- `results/tuning/`：参数搜索产生的图表和 CSV 明细。
- `results/final/`：最终测试、错分分析、耗时比较与重复验证的结果。
- `docs/`：实验报告、作业说明与实验日志。
- `archive/`：早期脚本和探索性图表，仅供追溯，不属于当前推荐流程。
- `data/`：教师提供的数据集；该目录不会提交到仓库。

## 环境与数据

推荐环境：Python 3.13.12、NumPy 1.26.4、PyTorch 2.12.0+cu126、Scikit-learn 1.8.0、Matplotlib 3.10.9。

将教师提供的 `training.pt` 与 `test.pt` 放在以下位置：

```text
data/processed_8_9/training.pt
data/processed_8_9/test.pt
```

## 推荐运行顺序

在项目根目录执行：

```powershell
python scripts/check_data.py
python scripts/run_grid_search.py --model ce
python scripts/final_evaluation.py
python scripts/repeated_validation.py
```

其中，`final_evaluation.py` 会使用已确定的最终超参数，训练四种最终模型并生成测试指标、损失曲线、耗时图、错分方向统计和全部错分样本图。它不会重复执行网格搜索。

如需单独搜索其他模型，可运行：

```powershell
python scripts/run_grid_search.py --model hinge
python scripts/run_grid_search.py --model linear_svm
python scripts/run_grid_search.py --model rbf_svm
```

Cross-Entropy 的学习率边界确认可运行：

```powershell
python scripts/run_grid_search.py --model ce --ce-final-check
```

如果仅需根据已有最终 CSV 重绘耗时图和错分数量图，可运行：

```powershell
python scripts/refresh_analysis_figures.py
```

## 说明

- 训练集和验证集使用分层划分；测试集仅用于最终一次评价。
- 多随机种子验证仅使用原始训练集，不读取测试集。
- `results/` 内的图表和 CSV 是实验报告中表格、插图与结论的可复核依据。

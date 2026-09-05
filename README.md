# RNN_LSTM_GRU Person Name Classifier Demo

> 这是一个用于学习和实验的 Demo，不是生产级人名识别系统。

使用字符级 RNN、LSTM 和 GRU，根据人名预测其所属国家类别，并记录三种模型的训练曲线、测试集准确率和混淆矩阵。

## 项目流程

```text
data/names/*.txt
        |
        v
读取、清洗、去重、构建字符表
        |
        v
字符级 one-hot 向量
        |
        v
RNN / LSTM / GRU 训练
        |
        v
验证集选取最佳权重，测试集进行最终评估
```

## 数据集

数据目录为 `data/names`，当前包含 18 个国家类别，每个 `.txt` 文件至少包含 200 个姓名，每行一个姓名。

文件名（不含扩展名）就是类别名，例如：

```text
data/names/French.txt  -> French
data/names/Chinese.txt -> Chinese
```

数据使用 UTF-8 编码。清洗过程会去除 BOM、首尾空白、控制字符、空行和重复姓名，同时保留重音符号、撇号、连字符、点号以及其他 Unicode 字符。

## 数据预处理

运行预处理脚本可以检查类别、样本数、字符表大小和向量形状：

```bash
python data_preprocessing.py
```

单个人名会被转换为形状为：

```text
(姓名长度, 1, 字符表大小)
```

其中每个字符使用一个 one-hot 向量表示，`1` 表示当前 batch 只有一个姓名。

## 模型训练

直接运行训练入口：

```bash
python train_models.py
```

默认设置如下：

- 训练集、验证集、测试集比例为 `70% / 15% / 15%`
- 训练集按类别划分，每类默认 140 条训练姓名
- 每个 epoch 完整遍历训练集一次
- 默认训练 20 个 epoch
- 隐藏层大小为 128
- 默认优化器为 Adam，学习率为 `0.001`
- 梯度裁剪阈值为 `5.0`

自定义训练量和超参数：

```bash
python train_models.py --epochs 10 --hidden-size 128 --learning-rate 0.001
```

只传入旧参数 `--iterations` 时，会使用兼容参考代码的随机抽样训练方式：

```bash
python train_models.py --iterations 2000
```

## 输出文件

每次训练会在 `artifacts/run_YYYYMMDD_HHMMSS/` 下生成结果：

- `training.log`：训练过程、epoch 验证准确率和最终测试准确率
- `training_history.json`：配置、数据集划分、逐步训练记录、测试指标和混淆矩阵
- `training_metrics.csv`：逐步 loss、准确率和预测类别
- `training_comparison.png`：训练 loss、训练准确率和测试准确率对比图
- `confusion_matrices.png`：RNN、LSTM、GRU 的测试集混淆矩阵
- `rnn_model.pt`、`lstm_model.pt`、`gru_model.pt`：验证集表现最佳的模型权重

混淆矩阵中，行表示真实类别，列表示预测类别。测试集只在训练结束后使用一次，不参与参数更新。

## Demo 结果示例

最近一次 10 epoch 训练使用了 2520 条训练姓名和 540 条测试姓名：

```text
RNN:  99.63% test accuracy
LSTM: 100.00% test accuracy
GRU:  100.00% test accuracy
```

结果目录：`artifacts/run_20260905_160108/`

以上结果仅用于展示 Demo 流程。当前数据集规模较小，且不同国家的文字系统差异较明显，准确率可能高于真实跨来源数据的泛化效果。正式系统还需要更大、更多来源的数据集、严格的数据去重和多次独立实验。

## 运行测试

```bash
python -m pytest -q
```

当前测试覆盖数据清洗、向量化、模型前向传播、训练步骤、数据集划分、详细评估、混淆矩阵和结果文件保存。

## 运行设备

当前版本未启用 GPU，训练使用 CPU。检查本机 PyTorch 是否能使用 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出 `False`，说明当前 PyTorch 或运行环境没有可用 CUDA。后续使用 GPU 时，需要安装 CUDA 版 PyTorch，并将模型、输入张量和标签迁移到同一个 CUDA 设备。

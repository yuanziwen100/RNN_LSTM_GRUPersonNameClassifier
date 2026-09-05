"""训练 RNN、LSTM 和 GRU 人名分类模型，并保存可比较的训练结果。

本模块对应参考代码中的第四步：

* 从模型输出得到预测类别；
* 随机生成训练样本；
* 分别训练传统 RNN、LSTM 和 GRU；
* 按类别划分训练集、验证集和测试集，并按 epoch 遍历训练集；
* 周期性打印 loss、准确率和样本预测结果；
* 保存每一步训练记录、模型权重、对比图和测试集混淆矩阵。

输入直接使用 :mod:`data_preprocessing` 生成的 one-hot 张量。模型输出是
``LogSoftmax`` 对数概率，因此损失函数使用 ``nn.NLLLoss``。
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

# 训练脚本通常在没有图形桌面的环境运行，使用无界面后端生成 PNG。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from data_preprocessing import NameDataset, load_name_data
from models import GRU, LSTM, MODEL_CLASSES, RNN


MODEL_NAMES = ("RNN", "LSTM", "GRU")


@dataclass(frozen=True)
class DatasetSplits:
    """按类别划分后的训练集、验证集和测试集。"""

    train: NameDataset
    validation: NameDataset
    test: NameDataset


@dataclass(frozen=True)
class EvaluationMetrics:
    """分类评估结果，包括总体准确率和混淆矩阵。"""

    accuracy: float
    confusion_matrix: List[List[int]]
    per_class_accuracy: Dict[str, float]
    total: int


@dataclass(frozen=True)
class TrainingConfig:
    """训练超参数和输出配置。"""

    iterations: int = 500
    # 新训练流程使用完整训练集遍历；保留 iterations 兼容旧的随机训练计划。
    epochs: Optional[int] = 20
    hidden_size: int = 128
    num_layers: int = 1
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    print_every: int = 50
    plot_every: int = 10
    gradient_clip: float = 5.0
    seed: Optional[int] = 42
    output_dir: Path = Path("artifacts")

    def __post_init__(self) -> None:
        """在训练开始前尽早发现无效配置。"""

        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.epochs is not None and self.epochs < 1:
            raise ValueError("epochs must be at least 1 when provided")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.optimizer not in {"SGD", "Adam"}:
            raise ValueError("optimizer must be either 'SGD' or 'Adam'")
        if self.print_every < 1 or self.plot_every < 1:
            raise ValueError("print_every and plot_every must be at least 1")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")


@dataclass
class ModelTrainingResult:
    """单个模型的训练记录、模型对象和最终评估结果。"""

    model_name: str
    model: nn.Module
    history: List[Dict[str, Any]]
    evaluation_accuracy: float
    elapsed_seconds: float
    validation_accuracy: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)

    @property
    def final_loss(self) -> float:
        """最后一步的训练 loss。"""

        return float(self.history[-1]["loss"])

    @property
    def final_training_accuracy(self) -> float:
        """到最后一步为止的累计训练准确率。"""

        return float(self.history[-1]["running_accuracy"])


def configure_training_logger(log_file: Optional[Path] = None) -> logging.Logger:
    """创建同时输出到控制台和可选日志文件的训练 logger。"""

    logger = logging.getLogger("person_name_classifier.training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 多次调用 run_training 时移除旧 handler，避免同一条日志重复打印。
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def time_since(start_time: float) -> str:
    """把计时器起点到当前的秒数格式化成 ``Xm Ys``。"""

    elapsed = max(0.0, time.perf_counter() - start_time)
    minutes, seconds = divmod(int(elapsed), 60)
    return f"{minutes}m {seconds}s"


def category_from_output(
    output: Tensor,
    all_categories: Sequence[str],
) -> Tuple[str, int]:
    """从模型输出的最大值获得类别名称和类别索引。

    模型输出可以是 ``(1, n_categories)`` 或 ``(n_categories,)``；两种形状都
    支持，便于训练和交互式推理时复用。
    """

    if output.ndim == 1:
        output = output.unsqueeze(0)
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("output must have shape (n_categories,) or (1, n_categories)")
    if output.shape[1] != len(all_categories):
        raise ValueError(
            f"output category size {output.shape[1]} does not match "
            f"number of categories {len(all_categories)}"
        )

    category_index = int(torch.argmax(output, dim=1).item())
    return all_categories[category_index], category_index


def random_training_example(
    dataset: NameDataset,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, Tensor, Tensor]:
    """随机抽取一条训练样本，返回类别、姓名、标签和 one-hot 输入。

    这是参考代码 ``randomTrainingExample`` 的 Python 版本。默认使用
    ``SystemRandom``，用于单独交互调用；正式比较训练时使用下面的固定训练计划，
    让三个模型看到完全相同的样本顺序。
    """

    random_source = rng or random.SystemRandom()
    category = random_source.choice(dataset.all_categories)
    name = random_source.choice(dataset.category_lines[category])
    target = dataset.category_to_tensor(category)
    input_tensor = dataset.line_to_tensor(name)
    return category, name, target, input_tensor


def build_training_schedule(
    dataset: NameDataset,
    iterations: int,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """生成供三个模型复用的随机训练样本计划。

    三个模型使用同一份计划，可以排除“某个模型刚好抽到更容易样本”造成的
    比较偏差。返回字典只保存字符串和整数，不保存 Tensor，便于写入 JSON。
    """

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    rng = random.Random(seed)
    category_indices = dataset.category_to_index
    schedule: List[Dict[str, Any]] = []
    for _ in range(iterations):
        category = rng.choice(dataset.all_categories)
        name = rng.choice(dataset.category_lines[category])
        schedule.append(
            {
                "category": category,
                "name": name,
                "target_index": category_indices[category],
            }
        )
    return schedule


def split_name_dataset(
    dataset: NameDataset,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: Optional[int] = 42,
) -> DatasetSplits:
    """按类别将数据划分为训练集、验证集和测试集。

    每个类别独立打乱和切分，因此类别比例保持一致。姓名只会出现在一个
    子集中，三份数据共用原始字符表，避免切分后出现输入维度不一致。
    """

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    rng = random.Random(seed)
    train_lines: Dict[str, List[str]] = {}
    validation_lines: Dict[str, List[str]] = {}
    test_lines: Dict[str, List[str]] = {}

    for category in dataset.all_categories:
        names = list(dataset.category_lines[category])
        if len(names) < 3:
            raise ValueError(
                f"category {category!r} needs at least 3 names to create three splits"
            )
        rng.shuffle(names)
        train_count = max(1, int(len(names) * train_ratio))
        validation_count = max(1, int(len(names) * validation_ratio))
        if train_count + validation_count >= len(names):
            validation_count = max(1, len(names) - train_count - 1)
        test_start = train_count + validation_count
        train_lines[category] = names[:train_count]
        validation_lines[category] = names[train_count:test_start]
        test_lines[category] = names[test_start:]

    categories = list(dataset.all_categories)
    return DatasetSplits(
        train=NameDataset(train_lines, categories, dataset.all_letters),
        validation=NameDataset(validation_lines, categories, dataset.all_letters),
        test=NameDataset(test_lines, categories, dataset.all_letters),
    )


def build_epoch_training_schedule(
    dataset: NameDataset,
    epochs: int,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """生成按 epoch 遍历训练集全部姓名的随机计划。

    每个 epoch 恰好包含训练集中的每条姓名一次，然后重新打乱顺序。三种模型
    复用同一份计划，可以在增加训练量的同时保持模型比较公平。
    """

    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    rng = random.Random(seed)
    category_indices = dataset.category_to_index
    examples = [
        (category, name)
        for category in dataset.all_categories
        for name in dataset.category_lines[category]
    ]
    schedule: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        epoch_examples = list(examples)
        rng.shuffle(epoch_examples)
        schedule.extend(
            {
                "epoch": epoch,
                "category": category,
                "name": name,
                "target_index": category_indices[category],
            }
            for category, name in epoch_examples
        )
    return schedule


def _forward_for_training(
    model: nn.Module,
    input_tensor: Tensor,
) -> Tensor:
    """按模型类型初始化状态并完成一次前向计算。"""

    # 每个人名都是独立序列，训练一个样本前将状态初始化为全零。
    if isinstance(model, LSTM):
        hidden, cell = model.init_hidden_and_cell(batch_size=input_tensor.shape[1])
        output, _, _ = model(input_tensor, hidden, cell)
    elif isinstance(model, (RNN, GRU)):
        hidden = model.init_hidden(batch_size=input_tensor.shape[1])
        output, _ = model(input_tensor, hidden)
    else:
        raise TypeError(f"unsupported model type: {type(model).__name__}")
    return output


def train_one_example(
    model: nn.Module,
    input_tensor: Tensor,
    target: Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    gradient_clip: float = 5.0,
) -> Tuple[float, int]:
    """训练一个人名样本，返回 ``(loss, predicted_index)``。"""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = _forward_for_training(model, input_tensor)
    loss = criterion(output, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("training loss became NaN or Inf")

    loss.backward()
    # 循环网络在较长序列上可能出现梯度爆炸，裁剪后再更新参数。
    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()

    predicted_index = int(torch.argmax(output.detach(), dim=1).item())
    return float(loss.detach().item()), predicted_index


def train_model(
    model_name: str,
    dataset: NameDataset,
    config: TrainingConfig,
    *,
    schedule: Optional[Sequence[Mapping[str, Any]]] = None,
    validation_dataset: Optional[NameDataset] = None,
    logger: Optional[logging.Logger] = None,
) -> ModelTrainingResult:
    """训练一个指定模型，并记录每个 iteration 的指标。"""

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"unknown model {model_name!r}; expected one of {MODEL_NAMES}")

    training_schedule = list(
        schedule
        if schedule is not None
        else build_training_schedule(dataset, config.iterations, config.seed)
    )
    if not training_schedule:
        raise ValueError("training schedule must contain at least one sample")
    if schedule is not None and len(training_schedule) < config.iterations:
        raise ValueError("training schedule is shorter than config.iterations")

    model_class = MODEL_CLASSES[model_name]
    model = model_class(
        input_size=dataset.n_letters,
        hidden_size=config.hidden_size,
        output_size=dataset.n_categories,
        num_layers=config.num_layers,
    )
    if config.optimizer == "Adam":
        # Adam 对字符级序列的梯度尺度更稳，默认用于本项目的实际训练；
        # 传入 optimizer="SGD" 仍可复现参考代码中的随机梯度下降方案。
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.NLLLoss()
    active_logger = logger or logging.getLogger("person_name_classifier.training")
    history: List[Dict[str, Any]] = []
    losses: List[float] = []
    correct_count = 0
    best_validation_accuracy = 0.0
    best_state_dict: Optional[Dict[str, Tensor]] = None
    start_time = time.perf_counter()
    total_steps = len(training_schedule)

    active_logger.info(
        "%s 开始训练: steps=%d epochs=%s hidden_size=%d optimizer=%s lr=%g",
        model_name,
        total_steps,
        config.epochs,
        config.hidden_size,
        config.optimizer,
        config.learning_rate,
    )

    for iteration in range(1, total_steps + 1):
        sample = training_schedule[iteration - 1]
        category = str(sample["category"])
        name = str(sample["name"])
        target_index = int(sample["target_index"])
        epoch_value = sample.get("epoch")
        input_tensor = dataset.line_to_tensor(name)
        target = torch.tensor([target_index], dtype=torch.long)

        loss, predicted_index = train_one_example(
            model,
            input_tensor,
            target,
            optimizer,
            criterion,
            gradient_clip=config.gradient_clip,
        )
        losses.append(loss)
        correct = int(predicted_index == target_index)
        correct_count += correct

        window_start = max(0, len(losses) - config.plot_every)
        window_losses = losses[window_start:]
        window_correct = sum(
            int(row["correct"])
            for row in history[window_start:]
        ) + correct
        running_accuracy = correct_count / iteration
        window_accuracy = window_correct / len(window_losses)
        validation_accuracy: Optional[float] = None
        is_epoch_end = (
            epoch_value is not None
            and (
                iteration == total_steps
                or training_schedule[iteration].get("epoch") != epoch_value
            )
        )
        if validation_dataset is not None and is_epoch_end:
            validation_accuracy = evaluate_model(model, validation_dataset)
            active_logger.info(
                "%s epoch %s/%s | validation_accuracy=%.2f%%",
                model_name,
                epoch_value,
                config.epochs if config.epochs is not None else "?",
                validation_accuracy * 100,
            )
            if best_state_dict is None or validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_state_dict = copy.deepcopy(model.state_dict())
        record = {
            "step": iteration,
            "loss": loss,
            "rolling_loss": sum(window_losses) / len(window_losses),
            "correct": correct,
            "running_accuracy": running_accuracy,
            "window_accuracy": window_accuracy,
            "category": category,
            "target_index": target_index,
            "name": name,
            "predicted_category": dataset.all_categories[predicted_index],
            "predicted_index": predicted_index,
            "epoch": int(epoch_value) if epoch_value is not None else None,
            "validation_accuracy": validation_accuracy,
        }
        history.append(record)

        if iteration == 1 or iteration % config.print_every == 0 or iteration == config.iterations:
            active_logger.info(
                "%s step %d/%d | loss=%.4f | avg_loss=%.4f | accuracy=%.2f%% "
                "| target=%s | predicted=%s | name=%r",
                model_name,
                iteration,
                total_steps,
                loss,
                record["rolling_loss"],
                running_accuracy * 100,
                category,
                dataset.all_categories[predicted_index],
                name,
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    elapsed_seconds = time.perf_counter() - start_time
    active_logger.info(
        "%s 训练完成: loss=%.4f train_accuracy=%.2f%% elapsed=%s",
        model_name,
        history[-1]["loss"],
        history[-1]["running_accuracy"] * 100,
        time_since(start_time),
    )
    return ModelTrainingResult(
        model_name=model_name,
        model=model,
        history=history,
        evaluation_accuracy=0.0,
        elapsed_seconds=elapsed_seconds,
        validation_accuracy=best_validation_accuracy,
    )


def evaluate_model(
    model: nn.Module,
    dataset: NameDataset,
    *,
    max_samples: Optional[int] = None,
) -> float:
    """在整个数据集（或指定数量样本）上计算分类准确率。"""

    return evaluate_model_details(model, dataset, max_samples=max_samples).accuracy


def evaluate_model_details(
    model: nn.Module,
    dataset: NameDataset,
    *,
    max_samples: Optional[int] = None,
) -> EvaluationMetrics:
    """评估模型并返回总体、分类型准确率和混淆矩阵。

    混淆矩阵的行是真实类别，列是预测类别；类别顺序与
    ``dataset.all_categories`` 一致。
    """

    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when provided")

    model.eval()
    correct = 0
    total = 0
    category_count = dataset.n_categories
    confusion_matrix = [
        [0 for _ in range(category_count)] for _ in range(category_count)
    ]
    category_correct = [0 for _ in range(category_count)]
    category_total = [0 for _ in range(category_count)]

    with torch.no_grad():
        for category in dataset.all_categories:
            target_index = dataset.category_to_index[category]
            names = dataset.category_lines[category]
            for name in names:
                if max_samples is not None and total >= max_samples:
                    break
                output = _forward_for_training(model, dataset.line_to_tensor(name))
                predicted_index = int(torch.argmax(output, dim=1).item())
                correct += int(predicted_index == target_index)
                total += 1
                confusion_matrix[target_index][predicted_index] += 1
                category_total[target_index] += 1
                category_correct[target_index] += int(predicted_index == target_index)
            if max_samples is not None and total >= max_samples:
                break

    per_class_accuracy = {
        category: (
            category_correct[index] / category_total[index]
            if category_total[index]
            else 0.0
        )
        for index, category in enumerate(dataset.all_categories)
    }
    return EvaluationMetrics(
        accuracy=correct / total if total else 0.0,
        confusion_matrix=confusion_matrix,
        per_class_accuracy=per_class_accuracy,
        total=total,
    )


def train_all_models(
    dataset: NameDataset,
    config: TrainingConfig,
    *,
    schedule: Optional[Sequence[Mapping[str, Any]]] = None,
    validation_dataset: Optional[NameDataset] = None,
    evaluation_dataset: Optional[NameDataset] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, ModelTrainingResult]:
    """使用同一训练计划训练三种模型，并在指定评估集上计算结果。"""

    active_logger = logger or logging.getLogger("person_name_classifier.training")
    shared_schedule = list(
        schedule
        if schedule is not None
        else (
            build_epoch_training_schedule(dataset, config.epochs, config.seed)
            if config.epochs is not None
            else build_training_schedule(dataset, config.iterations, config.seed)
        )
    )
    target_evaluation_dataset = evaluation_dataset or dataset
    results: Dict[str, ModelTrainingResult] = {}

    for model_index, model_name in enumerate(MODEL_NAMES):
        # 初始化种子不同，避免三个模型因参数初值完全相同而产生偶然耦合；
        # 训练样本顺序仍完全相同，比较重点保持公平。
        if config.seed is not None:
            torch.manual_seed(config.seed + model_index)
        result = train_model(
            model_name,
            dataset,
            config,
            schedule=shared_schedule,
            validation_dataset=validation_dataset,
            logger=active_logger,
        )
        evaluation = evaluate_model_details(result.model, target_evaluation_dataset)
        result.evaluation_accuracy = evaluation.accuracy
        result.confusion_matrix = evaluation.confusion_matrix
        result.per_class_accuracy = evaluation.per_class_accuracy
        active_logger.info(
            "%s 评估准确率: %.2f%% (%d samples)",
            model_name,
            result.evaluation_accuracy * 100,
            evaluation.total,
        )
        results[model_name] = result
    return results


def _json_config(config: TrainingConfig) -> Dict[str, Any]:
    """把 Path 等配置值转换成 JSON 可序列化类型。"""

    values = asdict(config)
    values["output_dir"] = str(values["output_dir"])
    return values


def save_training_results(
    results: Mapping[str, ModelTrainingResult],
    dataset: NameDataset,
    config: TrainingConfig,
    output_dir: Path,
    *,
    splits: Optional[DatasetSplits] = None,
) -> Dict[str, Any]:
    """保存 JSON、CSV、PNG 和三个模型 checkpoint，返回文件路径。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整历史，便于之后重新绘图或分析每一步的预测结果。
    summary = {
        "config": _json_config(config),
        "dataset": {
            "categories": dataset.all_categories,
            "n_categories": dataset.n_categories,
            "n_letters": dataset.n_letters,
            "n_samples": dataset.n_samples,
        },
        "splits": (
            {
                "train_samples": splits.train.n_samples,
                "validation_samples": splits.validation.n_samples,
                "test_samples": splits.test.n_samples,
            }
            if splits is not None
            else None
        ),
        "models": {
            model_name: {
                "final_loss": result.final_loss,
                "final_training_accuracy": result.final_training_accuracy,
                "validation_accuracy": result.validation_accuracy,
                "evaluation_accuracy": result.evaluation_accuracy,
                "test_accuracy": result.evaluation_accuracy,
                "confusion_matrix": result.confusion_matrix,
                "per_class_accuracy": result.per_class_accuracy,
                "elapsed_seconds": result.elapsed_seconds,
                "history": result.history,
            }
            for model_name, result in results.items()
        },
    }
    json_path = output_dir / "training_history.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "training_metrics.csv"
    fieldnames = [
        "model",
        "step",
        "loss",
        "rolling_loss",
        "correct",
        "running_accuracy",
        "window_accuracy",
        "epoch",
        "validation_accuracy",
        "category",
        "target_index",
        "name",
        "predicted_category",
        "predicted_index",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, result in results.items():
            for row in result.history:
                writer.writerow({"model": model_name, **row})

    checkpoint_paths: Dict[str, Path] = {}
    for model_name, result in results.items():
        checkpoint_path = output_dir / f"{model_name.lower()}_model.pt"
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": result.model.state_dict(),
                "input_size": dataset.n_letters,
                "hidden_size": config.hidden_size,
                "output_size": dataset.n_categories,
                "num_layers": config.num_layers,
                "all_categories": dataset.all_categories,
                "all_letters": dataset.all_letters,
            },
            checkpoint_path,
        )
        checkpoint_paths[model_name] = checkpoint_path

    plot_path = output_dir / "training_comparison.png"
    _save_comparison_plot(results, plot_path, config)
    confusion_matrix_path = output_dir / "confusion_matrices.png"
    _save_confusion_matrix_plot(results, dataset.all_categories, confusion_matrix_path)

    return {
        "json": json_path,
        "csv": csv_path,
        "plot": plot_path,
        "confusion_matrix": confusion_matrix_path,
        "checkpoints": checkpoint_paths,
    }


def _save_comparison_plot(
    results: Mapping[str, ModelTrainingResult],
    plot_path: Path,
    config: TrainingConfig,
) -> None:
    """绘制训练 loss、训练准确率和最终全量准确率对比图。"""

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = {"RNN": "#2563eb", "LSTM": "#dc2626", "GRU": "#059669"}

    for model_name, result in results.items():
        steps = [row["step"] for row in result.history]
        rolling_losses = [row["rolling_loss"] for row in result.history]
        running_accuracies = [row["running_accuracy"] * 100 for row in result.history]
        color = colors.get(model_name)
        axes[0].plot(steps, rolling_losses, label=model_name, color=color, linewidth=2)
        axes[1].plot(steps, running_accuracies, label=model_name, color=color, linewidth=2)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("NLL loss (rolling average)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    names = list(results)
    accuracies = [results[name].evaluation_accuracy * 100 for name in names]
    bars = axes[2].bar(
        names,
        accuracies,
        color=[colors.get(name) for name in names],
        width=0.6,
    )
    axes[2].set_title("Evaluation Accuracy")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_ylim(0, 100)
    axes[2].grid(axis="y", alpha=0.25)
    for bar, accuracy in zip(bars, accuracies):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            min(accuracy + 2, 98),
            f"{accuracy:.1f}%",
            ha="center",
        )

    total_steps = max((len(result.history) for result in results.values()), default=0)
    training_label = (
        f"{config.epochs} epochs | {total_steps} steps"
        if config.epochs is not None
        else f"{config.iterations} iterations"
    )
    figure.suptitle(
        f"RNN vs LSTM vs GRU | {training_label} | hidden={config.hidden_size}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_confusion_matrix_plot(
    results: Mapping[str, ModelTrainingResult],
    categories: Sequence[str],
    plot_path: Path,
) -> None:
    """绘制各模型测试集混淆矩阵，行是真实类别、列是预测类别。"""

    model_names = list(results)
    figure, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(8 * len(model_names), 7),
        squeeze=False,
    )
    axes_row = axes[0]
    category_indices = range(len(categories))

    for axis, model_name in zip(axes_row, model_names):
        matrix = results[model_name].confusion_matrix
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set_title(model_name)
        axis.set_xlabel("Predicted category")
        axis.set_ylabel("True category")
        axis.set_xticks(list(category_indices))
        axis.set_yticks(list(category_indices))
        axis.set_xticklabels(categories, rotation=90, fontsize=7)
        axis.set_yticklabels(categories, fontsize=7)
        threshold = max((max(row) for row in matrix), default=0) / 2
        for row_index, row in enumerate(matrix):
            for column_index, count in enumerate(row):
                axis.text(
                    column_index,
                    row_index,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if count > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("Test Set Confusion Matrices")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_training(
    config: Optional[TrainingConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> Tuple[NameDataset, Dict[str, ModelTrainingResult], Dict[str, Any]]:
    """执行完整训练流程并保存结果。"""

    active_config = config or TrainingConfig()
    result_dir = Path(output_dir or active_config.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(result_dir / "training.log")

    if active_config.seed is not None:
        random.seed(active_config.seed)
        torch.manual_seed(active_config.seed)

    logger.info("读取数据: %s", load_name_data.__name__)
    dataset = load_name_data(min_names_per_category=200)
    splits = split_name_dataset(dataset, seed=active_config.seed)
    logger.info(
        "数据集: categories=%d samples=%d letters=%d | train=%d validation=%d test=%d",
        dataset.n_categories,
        dataset.n_samples,
        dataset.n_letters,
        splits.train.n_samples,
        splits.validation.n_samples,
        splits.test.n_samples,
    )
    if active_config.epochs is not None:
        schedule = build_epoch_training_schedule(
            splits.train,
            active_config.epochs,
            active_config.seed,
        )
    else:
        schedule = build_training_schedule(
            splits.train,
            active_config.iterations,
            active_config.seed,
        )
    results = train_all_models(
        splits.train,
        active_config,
        schedule=schedule,
        validation_dataset=splits.validation,
        evaluation_dataset=splits.test,
        logger=logger,
    )
    paths = save_training_results(
        results,
        dataset,
        active_config,
        result_dir,
        splits=splits,
    )
    logger.info("训练记录已保存: %s", result_dir.resolve())
    logger.info("效果图: %s", paths["plot"].resolve())
    logger.info("混淆矩阵: %s", paths["confusion_matrix"].resolve())
    return dataset, results, paths


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Train and compare RNN, LSTM, and GRU name classifiers"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="legacy random-sampling update count; used when --epochs is omitted",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="number of complete passes over the training split (default: 20)",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="Adam")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--plot-every", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def main() -> int:
    """命令行入口：创建带时间戳的运行目录并启动三模型训练。"""

    # Windows 控制台可能仍使用本地代码页；训练姓名和类别包含多语言字符，
    # 统一切换到 UTF-8，避免终端日志乱码。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _build_argument_parser().parse_args()
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    if args.epochs is not None:
        iterations = args.iterations or 500
        epochs = args.epochs
    elif args.iterations is not None:
        iterations = args.iterations
        epochs = None
    else:
        iterations = 500
        epochs = 20

    config = TrainingConfig(
        iterations=iterations,
        epochs=epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        print_every=args.print_every,
        plot_every=args.plot_every,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        output_dir=output_dir,
    )
    dataset, results, paths = run_training(config, output_dir=output_dir)

    print("\n训练摘要")
    print(f"类别数: {dataset.n_categories}，样本数: {dataset.n_samples}")
    for model_name, result in results.items():
        print(
            f"{model_name}: final_loss={result.final_loss:.4f}, "
            f"validation_accuracy={result.validation_accuracy * 100:.2f}%, "
            f"test_accuracy={result.evaluation_accuracy * 100:.2f}%"
        )
    print(f"效果图: {paths['plot'].resolve()}")
    print(f"混淆矩阵: {paths['confusion_matrix'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

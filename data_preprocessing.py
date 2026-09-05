"""人名分类项目的数据读取、清洗和向量化工具。

1. 使用 Unicode NFC 规范化。中文、阿拉伯文、
   希腊文以及带重音的拉丁字母都是有效的训练输入。
2. 清洗时保留 Unicode 字母、组合重音符号和 Unicode 标点。这样 ``Émile``、
   ``d'Aubigné``、``Jean-Luc``、``J.-P.`` 等写法不会因为预处理而丢失信息。

典型用法::

    from data_preprocessing import load_name_data, vectorize_dataset

    dataset = load_name_data()  # 默认读取项目根目录下的 data/names
    inputs, targets = vectorize_dataset(dataset)
    print(inputs[0].shape, targets.shape)

其中每个人名对应一个形状为 ``(序列长度, 1, 字符表大小)`` 的 one-hot 张量，
可以直接作为 RNN、LSTM 或 GRU 的单条序列输入。
"""

from __future__ import annotations

import argparse
import glob
import os
import string
import sys
import unicodedata
from dataclasses import dataclass
from io import open
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


# 让默认路径不依赖运行命令时所在的当前目录。无论从项目根目录还是 IDE 启动，
# 都会定位到同一个 data/names 文件夹。
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "names"

# 参考代码把 ASCII 字母放在字符表前面。这里继续沿用这个顺序，然后再追加
# 数据集中出现的 Unicode 字符，使字符表既稳定又容易查看。
ASCII_PRIORITY = string.ascii_letters + " .,;'-"


def clean_name(name: str) -> str:
    """清洗单个人名，同时保留有效的多语言字符。

    清洗规则：

    * NFC 规范化会把 ``e`` 加组合重音整理成单个 ``é``，但不会抹掉重音；
    * 去除 UTF-8 BOM、首尾空白和控制字符；
    * 连续空白（空格、制表符、不换行空格等）统一为一个普通空格；
    * 保留 Unicode 字母（``L``）、组合标记（``M``）和标点（``P``）。

    参数
    ----
    name:
        原始文本中的一行人名。

    返回
    ------
    str
        清洗后的非空人名。

    异常
    ------
    TypeError
        ``name`` 不是字符串时抛出。
    """

    if not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name).__name__}")

    # NFC 只做规范化，不进行 ASCII 化，因而可以保留各语言中的真实字符。
    normalized = unicodedata.normalize("NFC", name).replace("\ufeff", "")

    cleaned: List[str] = []
    pending_space = False
    for char in normalized:
        if char.isspace():
            # 先记住空白，等遇到下一个有效字符时再决定是否写入一个空格。
            pending_space = bool(cleaned)
            continue

        category = unicodedata.category(char)
        is_name_character = (
            char.isalpha()
            or category.startswith("M")  # 组合重音等 Unicode 标记
            or category.startswith("P")  # 撇号、连字符、句点等 Unicode 标点
        )
        if not is_name_character:
            # 数字、控制字符和其它无关符号不进入字符表。
            continue

        if pending_space and cleaned:
            cleaned.append(" ")
        cleaned.append(char)
        pending_space = False

    return "".join(cleaned).strip()


def read_name_lines(filename: os.PathLike[str] | str, deduplicate: bool = True) -> List[str]:
    """从 UTF-8 文本文件读取并清洗人名。

    每一行视为一个样本。空行和清洗后为空的行会被跳过；默认还会删除重复项，
    并保持第一次出现的顺序，避免同一个样本无意中重复进入训练集。
    """

    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"name file does not exist: {path}")

    names: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            name = clean_name(raw_line.rstrip("\r\n"))
            if not name:
                continue
            if deduplicate and name in seen:
                continue
            names.append(name)
            seen.add(name)
    return names


def build_alphabet(category_lines: Dict[str, Sequence[str]]) -> str:
    """根据全部人名构造稳定的字符表。

    字符表是后续 one-hot 编码的列定义。ASCII 字母和常见标点优先排列，
    其余 Unicode 字符按码点排序，确保同一份数据每次运行都得到相同的索引。
    """

    all_characters = {char for names in category_lines.values() for name in names for char in name}
    priority_characters = [char for char in ASCII_PRIORITY if char in all_characters]
    priority_set = set(priority_characters)
    unicode_characters = sorted(
        all_characters - priority_set,
        key=lambda char: (unicodedata.name(char, ""), ord(char)),
    )
    alphabet = "".join(priority_characters + unicode_characters)
    if not alphabet:
        raise ValueError("cannot build an alphabet from empty name data")
    return alphabet


@dataclass
class NameDataset:
    """已加载的人名数据及其字符、类别索引。"""

    category_lines: Dict[str, List[str]]
    all_categories: List[str]
    all_letters: str

    @property
    def n_categories(self) -> int:
        """类别数量，作为分类器输出层的大小。"""

        return len(self.all_categories)

    @property
    def n_letters(self) -> int:
        """字符表大小，作为 one-hot 向量的最后一维。"""

        return len(self.all_letters)

    @property
    def category_to_index(self) -> Dict[str, int]:
        """类别名称到整数标签的映射。"""

        return {category: index for index, category in enumerate(self.all_categories)}

    @property
    def n_samples(self) -> int:
        """数据集中人名样本总数。"""

        return sum(len(names) for names in self.category_lines.values())

    def line_to_tensor(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """使用当前数据集字符表，将一个人名转换为 one-hot 张量。"""

        return line_to_tensor(name, self.all_letters, dtype=dtype)

    def category_to_tensor(self, category: str) -> torch.Tensor:
        """将类别名称转换为交叉熵损失可用的整数标签张量。"""

        return category_to_tensor(category, self.all_categories)


def load_name_data(
    data_path: os.PathLike[str] | str | None = None,
    *,
    min_names_per_category: int = 1,
) -> NameDataset:
    """扫描 ``*.txt`` 文件并建立类别到人名的映射。

    文件名（不含扩展名）就是类别名，例如 ``French.txt`` 对应 ``French``。
    路径按文件名排序，所以类别标签在不同机器和不同运行之间保持一致。

    ``min_names_per_category`` 默认设为 1，方便对临时小样本做单元测试；项目正式
    数据可传入 200，强制检查每一类满足本项目的数据集要求。
    """

    if min_names_per_category < 1:
        raise ValueError("min_names_per_category must be at least 1")

    directory = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
    if not directory.is_dir():
        raise FileNotFoundError(f"name data directory does not exist: {directory}")

    # 使用 glob 找到所有文本文件，与参考代码的 glob(data_path + '*.txt') 思路一致。
    filenames = sorted(glob.glob(str(directory / "*.txt")), key=lambda item: Path(item).name.casefold())
    if not filenames:
        raise FileNotFoundError(f"no .txt name files found in: {directory}")

    category_lines: Dict[str, List[str]] = {}
    for filename in filenames:
        category = Path(filename).stem
        if category in category_lines:
            raise ValueError(f"duplicate category name: {category}")
        names = read_name_lines(filename)
        if len(names) < min_names_per_category:
            raise ValueError(
                f"category {category!r} has {len(names)} names; "
                f"expected at least {min_names_per_category}"
            )
        category_lines[category] = names

    all_categories = list(category_lines)
    all_letters = build_alphabet(category_lines)
    return NameDataset(category_lines, all_categories, all_letters)


def category_to_tensor(category: str, all_categories: Sequence[str]) -> torch.Tensor:
    """把类别名称编码为形状 ``(1,)`` 的 ``torch.long`` 标签。"""

    try:
        category_index = all_categories.index(category)
    except ValueError as exc:
        raise ValueError(f"unknown category {category!r}; expected one of {list(all_categories)!r}") from exc
    return torch.tensor([category_index], dtype=torch.long)


def line_to_tensor(
    line: str,
    all_letters: str,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """将一个人名转换为字符级 one-hot 张量。

    返回张量的形状是 ``(len(line), 1, n_letters)``：

    * 第 0 维是时间步/字符位置；
    * 第 1 维是 batch 维，单条样本固定为 1；
    * 第 2 维是字符表，每个位置只有一个 1，其余为 0。

    这种布局正好适合后续按时间步喂给 ``nn.RNN``、``nn.LSTM`` 或 ``nn.GRU``。
    """

    if not isinstance(all_letters, str) or not all_letters:
        raise ValueError("all_letters must be a non-empty string")
    if len(set(all_letters)) != len(all_letters):
        raise ValueError("all_letters must not contain duplicate characters")

    normalized_line = clean_name(line)
    if not normalized_line:
        raise ValueError("cannot vectorize an empty name")

    unknown_characters = sorted(set(normalized_line) - set(all_letters), key=ord)
    if unknown_characters:
        printable = ", ".join(repr(char) for char in unknown_characters)
        raise ValueError(f"name contains characters absent from all_letters: {printable}")

    tensor = torch.zeros((len(normalized_line), 1, len(all_letters)), dtype=dtype, device=device)
    for time_step, character in enumerate(normalized_line):
        tensor[time_step, 0, all_letters.index(character)] = 1.0
    return tensor


def vectorize_dataset(dataset: NameDataset) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """把整个数据集转换成模型可用的输入列表和标签张量。

    人名长度不同，不能直接 ``torch.stack`` 成同一个无 padding 的三维张量，
    因此输入保留为 one-hot 张量列表；每个元素形状仍为
    ``(序列长度, 1, n_letters)``。标签可以直接传给 ``nn.CrossEntropyLoss``。
    """

    inputs: List[torch.Tensor] = []
    labels: List[int] = []
    category_indices = dataset.category_to_index
    for category in dataset.all_categories:
        for name in dataset.category_lines[category]:
            inputs.append(line_to_tensor(name, dataset.all_letters))
            labels.append(category_indices[category])

    return inputs, torch.tensor(labels, dtype=torch.long)


def _build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数，便于不打开 Notebook 也能检查预处理结果。"""

    parser = argparse.ArgumentParser(description="Load, clean, and vectorize multilingual names")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="name data directory (default: %(default)s)",
    )
    parser.add_argument(
        "--min-names",
        type=int,
        default=200,
        help="minimum number of names required for every category (default: %(default)s)",
    )
    return parser


def main() -> None:
    """命令行入口：读取数据并打印字符表、类别和向量维度摘要。"""

    # Windows 控制台可能仍使用 GBK；输出多语言样本前切换到 UTF-8，避免
    # 打印阿拉伯文、希腊文或中文时触发 UnicodeEncodeError。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _build_argument_parser().parse_args()
    dataset = load_name_data(args.data_path, min_names_per_category=args.min_names)
    inputs, labels = vectorize_dataset(dataset)

    print(f"data_path: {Path(args.data_path).resolve()}")
    print(f"categories ({dataset.n_categories}): {dataset.all_categories}")
    print(f"samples: {dataset.n_samples}")
    print(f"n_letters: {dataset.n_letters}")
    print(f"first_name: {dataset.category_lines[dataset.all_categories[0]][0]!r}")
    print(f"first_tensor_shape: {tuple(inputs[0].shape)}")
    print(f"labels_shape: {tuple(labels.shape)}")


if __name__ == "__main__":
    main()

"""用于人名分类的 RNN、LSTM 和 GRU 模型。

输入格式与 :func:`data_preprocessing.line_to_tensor` 保持一致：

    (sequence_length, batch_size, input_size)

其中 ``input_size`` 就是字符表大小。三个模型都使用最后一个时间步的隐藏
表示进行分类，并返回对数概率，适合配合 ``torch.nn.NLLLoss`` 使用。

示例::

    from data_preprocessing import load_name_data
    from models import RNN

    dataset = load_name_data()
    model = RNN(dataset.n_letters, hidden_size=128,
                output_size=dataset.n_categories)
    hidden = model.init_hidden()
    log_probs, hidden = model(dataset.line_to_tensor("Arthur Martin"), hidden)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def _validate_model_parameters(
    input_size: int,
    hidden_size: int,
    output_size: int,
    num_layers: int,
    dropout: float,
) -> None:
    """统一检查三个循环模型的构造参数。"""

    for name, value in (
        ("input_size", input_size),
        ("hidden_size", hidden_size),
        ("output_size", output_size),
        ("num_layers", num_layers),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")


class _NameClassifierBase(nn.Module):
    """三个模型共享的输入整理、分类头和隐藏状态工具。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_model_parameters(
            input_size, hidden_size, output_size, num_layers, dropout
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = float(dropout)

        # 参考代码通过 Linear + LogSoftmax 输出分类结果。
        self.linear = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    @staticmethod
    def _prepare_input(input_tensor: Tensor) -> Tensor:
        """把单条序列整理成 PyTorch 默认的三维 seq-first 格式。

        ``line_to_tensor`` 已经返回三维张量；为了方便调试，这里也接受二维的
        ``(sequence_length, input_size)`` 输入，并自动补充 batch 维度。
        """

        if not isinstance(input_tensor, Tensor):
            raise TypeError(
                f"input must be a torch.Tensor, got {type(input_tensor).__name__}"
            )
        if input_tensor.ndim == 2:
            input_tensor = input_tensor.unsqueeze(1)
        if input_tensor.ndim != 3:
            raise ValueError(
                "input must have shape (sequence_length, batch_size, input_size) "
                "or (sequence_length, input_size)"
            )
        if input_tensor.shape[0] < 1 or input_tensor.shape[1] < 1 or input_tensor.shape[2] < 1:
            raise ValueError("input dimensions must all be positive")
        return input_tensor

    def _classify_last_step(self, sequence_output: Tensor) -> Tensor:
        """取最后时间步输出，映射到类别并转换为对数概率。"""

        # sequence_output: (sequence_length, batch_size, hidden_size)
        last_step = sequence_output[-1]
        return self.log_softmax(self.linear(last_step))

    def _check_input_size(self, input_tensor: Tensor) -> None:
        """检查输入最后一维是否与模型字符表大小一致。"""

        if input_tensor.shape[2] != self.input_size:
            raise ValueError(
                f"input feature size {input_tensor.shape[2]} does not match "
                f"model input_size {self.input_size}"
            )

    def _initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """按模型参数的设备和类型创建全零隐藏状态。"""

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

        parameter = next(self.parameters())
        state_device = parameter.device if device is None else device
        state_dtype = parameter.dtype if dtype is None else dtype
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=state_device,
            dtype=state_dtype,
        )


class RNN(_NameClassifierBase):
    """传统循环神经网络人名分类器。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout)
        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError("nonlinearity must be either 'tanh' or 'relu'")

        # 单层循环网络传入 dropout 没有意义，PyTorch 也会发出警告，因此只在
        # 多层网络时启用层间 dropout。
        effective_dropout = self.dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            dropout=effective_dropout,
        )

    def forward(
        self,
        input_tensor: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """完成一次前向计算，返回 ``(log_probs, next_hidden)``。"""

        input_tensor = self._prepare_input(input_tensor)
        self._check_input_size(input_tensor)
        batch_size = input_tensor.shape[1]
        if hidden is None:
            hidden = self.init_hidden(
                batch_size=batch_size,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        sequence_output, next_hidden = self.rnn(input_tensor, hidden)
        return self._classify_last_step(sequence_output), next_hidden

    def init_hidden(
        self,
        batch_size: int = 1,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """初始化形状为 ``(num_layers, batch_size, hidden_size)`` 的隐藏状态。"""

        return self._initial_state(batch_size, device=device, dtype=dtype)

    # 参考图片使用 initHidden 命名；保留别名方便迁移旧的 Notebook 代码。
    def initHidden(self) -> Tensor:
        """兼容参考代码的单样本隐藏状态初始化方法。"""

        return self.init_hidden()


class LSTM(_NameClassifierBase):
    """长短期记忆网络人名分类器。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout)
        effective_dropout = self.dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
        )

    def forward(
        self,
        input_tensor: Tensor,
        hidden: Optional[Tensor | Tuple[Tensor, Tensor]] = None,
        cell: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """返回 ``(log_probs, next_hidden, next_cell)``。

        为兼容常见 PyTorch 写法，``hidden`` 既可传隐藏状态，也可传
        ``(hidden, cell)`` 二元组；参考图片中的三个参数写法同样支持。
        """

        input_tensor = self._prepare_input(input_tensor)
        self._check_input_size(input_tensor)
        batch_size = input_tensor.shape[1]

        if isinstance(hidden, tuple):
            if len(hidden) != 2:
                raise ValueError("LSTM hidden tuple must contain (hidden, cell)")
            if cell is not None:
                raise ValueError("pass cell either in hidden tuple or as cell, not both")
            hidden, cell = hidden

        if hidden is None and cell is None:
            hidden, cell = self.init_hidden_and_cell(
                batch_size=batch_size,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
        elif hidden is None or cell is None:
            raise ValueError("hidden and cell must be provided together")

        sequence_output, (next_hidden, next_cell) = self.lstm(
            input_tensor, (hidden, cell)
        )
        return self._classify_last_step(sequence_output), next_hidden, next_cell

    def init_hidden_and_cell(
        self,
        batch_size: int = 1,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor]:
        """同时初始化 LSTM 所需的 hidden state 和 cell state。"""

        state = self._initial_state(batch_size, device=device, dtype=dtype)
        cell = torch.zeros_like(state)
        return state, cell

    def initHiddenAndC(self) -> Tuple[Tensor, Tensor]:
        """兼容参考代码的单样本初始化方法。"""

        return self.init_hidden_and_cell()


class GRU(_NameClassifierBase):
    """门控循环单元网络人名分类器。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_size, hidden_size, output_size, num_layers, dropout)
        effective_dropout = self.dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
        )

    def forward(
        self,
        input_tensor: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """完成一次前向计算，返回 ``(log_probs, next_hidden)``。"""

        input_tensor = self._prepare_input(input_tensor)
        self._check_input_size(input_tensor)
        batch_size = input_tensor.shape[1]
        if hidden is None:
            hidden = self.init_hidden(
                batch_size=batch_size,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        sequence_output, next_hidden = self.gru(input_tensor, hidden)
        return self._classify_last_step(sequence_output), next_hidden

    def init_hidden(
        self,
        batch_size: int = 1,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """初始化形状为 ``(num_layers, batch_size, hidden_size)`` 的隐藏状态。"""

        return self._initial_state(batch_size, device=device, dtype=dtype)

    def initHidden(self) -> Tensor:
        """兼容参考代码的单样本隐藏状态初始化方法。"""

        return self.init_hidden()


# 训练脚本可以使用这个映射统一构造三种模型。
MODEL_CLASSES = {"RNN": RNN, "LSTM": LSTM, "GRU": GRU}


__all__ = ["GRU", "LSTM", "MODEL_CLASSES", "RNN"]

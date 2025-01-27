from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

from ...dataset.types import BatchedViews, DataShim
from ..types import Gaussians

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    """
    Encoder 是一个抽象基类，继承自 PyTorch 的 nn.Module 和 Python 标准库中的 ABC（Abstract Base Class）
    它的主要作用是定义一种通用的编码器接口，用于实现对数据的特定编码逻辑。
    """
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        deterministic: bool,
    ) -> Gaussians:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x

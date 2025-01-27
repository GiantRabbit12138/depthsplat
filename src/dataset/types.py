# 定义了一组与数据处理相关的 类型注释 和 数据结构，主要用于 PyTorch 和 jaxtyping

from typing import Callable, Literal, TypedDict

from jaxtyping import Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]


# The following types mainly exist to make type-hinted keys show up in VS Code. Some
# dimensions are annotated as "_" because either:
# 1. They're expected to change as part of a function call (e.g., resizing the dataset).
# 2. They're expected to vary within the same function call (e.g., the number of views,
#    which differs between context and target BatchedViews).


class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view


class BatchedExample(TypedDict, total=False):
    target: BatchedViews
    context: BatchedViews
    scene: list[str]


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    image: Float[Tensor, "_ 3 height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str


# A data shim modifies the example after it's been returned from the data loader.
# DataShim 是一个 类型别名，使用 Callable 定义了一种函数的类型签名
# 它规定 任何被称为 DataShim 的对象都必须是一个函数，并且符合以下规则:
# 输入参数类型是 BatchedExample
# 返回值类型也是 BatchedExample
DataShim = Callable[[BatchedExample], BatchedExample]

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews

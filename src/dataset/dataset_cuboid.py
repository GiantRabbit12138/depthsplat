import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

from ..misc.image_io import save_image

try:
    import os
    import time
    import matplotlib.pyplot as plt
    from matplotlib import cm
except ImportError:
    raise ImportError("matplotlib is required for cuboid dataset")

try:
    from utils.depth_image_processor import DepthImageProcessor
except ImportError:
    raise ImportError("DepthImageProcessor is required for cuboid dataset")


def save_color_map(np_image, save_path):
    """
    保存伪彩色图像
    """
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(np_image, cmap='turbo')  # 使用伪彩色 'turbo' 显示
    plt.colorbar()  # 添加颜色条用于参考
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@dataclass
class DatasetCuboidCfg(DatasetCfgCommon):
    """
    @dataclass 的主要作用是减少编写“数据类”时的样板代码。
    数据类通常是那些主要用于存储数据的类，它们不包含复杂的方法逻辑。
    通过使用 @dataclass，可以只定义类的字段，而不需要显式地编写初始化方法或其他特殊方法。
    """
    name: Literal["cuboid"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False
    use_index_to_load_chunk: Optional[bool] = False


class DatasetCuboid(IterableDataset):
    """get_dataset函数返回的对象"""
    cfg: DatasetCuboidCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetCuboidCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        # train模式下 self.chunks下保存了所有的.torch文件路径
        self.chunks = []
        for i, root in enumerate(cfg.roots):
            # 得到data_stage(train或者是test)下的文件夹路径
            root = root / self.data_stage
            # use_index_to_load_chunk == false 表示使用index.json文件
            if self.cfg.use_index_to_load_chunk:
                with open(root / "index.json", "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(list(set(json_dict.values())))
            else:
                # 获取后缀名为.torch的文件并对最终的list进行排序 结果放在root_chunks
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.suffix == ".torch"]
                )
                # debug
                print(f"root_chunks: {root_chunks}")
            # 将结果放入self.chunks中
            self.chunks.extend(root_chunks)
        # overfit_to_scene == null
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # testing on a subset for fast speed
            self.chunks = self.chunks[::cfg.test_chunk_interval]
        
        # 对数据集中的所有深度图进行归一化
        # 遍历所有的torch文件路径
        for chunk_path in self.chunks:
            print(f"chunk_path: {chunk_path}")
            # Load the chunk.
            chunk = torch.load(chunk_path)
            print(f"len(chunk): {len(chunk)}")
            for idx in range(len(chunk)):
                example = chunk[idx]
                depth_images = example["depth_images"]
                
                # debug
                # print(f"type(depth_images): {type(depth_images)}")
                
                # 找到所有图片中的灰度最小值和最大值
                min_val = np.inf
                max_val = -np.inf
                np_images_list = []
                # save_path = './depth_images_processed_invalid_depth_outputs'
                # os.makedirs(save_path, exist_ok=True)  # 创建保存目录
                for idx, image in enumerate(depth_images):
                    # debug
                    # print(f"image.shape: {image.shape}")
                    # 打开并加载图像
                    image = Image.open(BytesIO(image.numpy().tobytes()))
                    np_image = np.array(image, dtype=np.float32)  # 转为 NumPy 数组
                    np_images_list.append(np_image)
                    
                    # -----------测试 DepthImageProcessor 类-----------
                    # 初始化深度图像处理器
                    processor = DepthImageProcessor(threshold_ratio=0.8, bins=256)
                    
                    # # 分析直方图并保存
                    # hist, bin_edges = processor.analyze_histogram(
                    #     image, save_path=f"./histograms/hist_{idx:05}.png")
                    
                    # 丢弃掉直方图中的无效区间
                    np_image = processor.discard_depth_ranges(
                        np_image, [(10000, 65535)]
                    )
                    # # 保存伪彩色图像
                    # save_path = './depth_images_processed_invalid_depth_outputs'
                    # os.makedirs(save_path, exist_ok=True)  # 创建保存目录
                    # save_color_map(
                    #     processed_depth_image, 
                    #     os.path.join(save_path,  f"processed_depth_{idx:05}.png")
                    #     )
                    # ------------------------------------------------

                    # 将 I;16 图像转换为 NumPy 数组并归一化
                    if image.mode == "I;16":
                        if np_image.max() > max_val:
                            max_val = np_image.max()
                        if np_image.min() < min_val:
                            min_val = np_image.min()
                        # # 保存伪彩色图像
                        # save_color_map(image, os.path.join(save_path, f"depth_{idx:05}.png"))
                    else:
                        raise ValueError(f"Unsupported image mode: {image.mode}")
                print(f"min_val: {min_val}, max_val: {max_val}")
                self.depth_min_val = min_val
                self.depth_max_val = max_val
                # print(f"len(np_images_list): {len(np_images_list)}")
                
                # debug 对每个深度图进行归一化
                # np_images_normalized_list = []
                # save_path = './depth_images_normalized_outputs'
                # os.makedirs(save_path, exist_ok=True)  # 创建保存目录
                # for idx, img in enumerate(np_images_list):
                #     img_normalized = (img - self.depth_min_val) / (self.depth_max_val - self.depth_min_val)
                #     np_images_normalized_list.append(img_normalized)
                #     # torch_image_normalized = torch.from_numpy(img_normalized)
                #     # 保存伪彩色图像
                #     save_color_map(img_normalized, os.path.join(save_path, f"nomalized_{idx:05}.png"))
                # print(f"np_images_normalized_list: {len(np_images_normalized_list)}")
                

    def shuffle(self, lst: list) -> list:
        # 生成一个随机排列的整数序列 长度为 len(lst) 返回的是一个包含 0 到 len(lst)-1 的随机排列的张量
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    # DatasetCuboid继承自IterableDataset 因此必须要实现__iter__
    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]
        
        # 遍历所有的torch文件路径
        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)
            # debug
            # print(f"chunk:\n{chunk}")

            # overfit_to_scene == null
            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            # 如果shuffle_val==true 则train或者val模式下都随机打乱chunk 否则只有train模式下才打乱chunk
            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)
                # debug
                # print(f"after shuffle, chunk:\n{chunk}")

            # train_times_per_scene == 1
            times_per_scene = (
                1
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            # debug
            # print(f"chunk len:{len(chunk)}")
            # len(chunk) = 1
            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)
                # debug context_images.shape: torch.Size([2, 3, 180, 320])
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)
                # debug target_images.shape: torch.Size([1, 3, 180, 320])
                
                # 加载深度图
                context_depth_images = [
                    example["depth_images"][index.item()] for index in context_indices
                ]
                # print(f"context_depth_images:\n{context_depth_images}")
                context_depth_images = self.convert_depth_images(context_depth_images)
                target_depth_images = [
                    example["depth_images"][index.item()] for index in target_indices
                ]
                target_depth_images = self.convert_depth_images(target_depth_images)
                # debug
                # print(f"context_depth_images.shape:{context_depth_images.shape}")
                # print(f"target_depth_images.shape:{target_depth_images.shape}")

                # Skip the example if the images don't have the right shape.
                if self.cfg.highres:
                    expected_shape = (3, 720, 1280)
                else:
                    # expected_shape = (3, 360, 640)
                    # 这里改为了images_4里面的图片尺寸
                    expected_shape = (3, 180, 320)
                context_image_invalid = context_images.shape[1:] != expected_shape
                target_image_invalid = target_images.shape[1:] != expected_shape
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                nf_scale = 1.0
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "depth_image": context_depth_images,
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "depth_image": target_depth_images,
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "index": target_indices,
                    },
                    "scene": scene,
                }

                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        """
        poses: 输入张量，形状为[batch，18] 内参：焦距（fx，fy）主点偏移（cx，cy） 外参：w2c的旋转和平移向量
        返回：一个元组，包含两个张量外参矩阵（extrinsics），形状为[batch，4，4] 内参矩阵（intrinsics），形状为[batch，3，3]
        """
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    
    
    def convert_depth_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 1 height width"]:
        """
        加载深度图->对深度图进行归一化->转换为PyTorch张量[batch 1 height width]
        """
        torch_images = []
        for image in images:
            # 打开并加载图像
            image = Image.open(BytesIO(image.numpy().tobytes()))
            # print(f"image:\n{image}")

            # 将 I;16 图像转换为 NumPy 数组并归一化
            if image.mode == "I;16":
                np_image = np.array(image, dtype=np.float32)  # 转为 NumPy 数组
                # 动态归一化到 [0, 1]，根据最大值和最小值
                max_val = self.depth_max_val
                min_val = self.depth_min_val
                # print(f"max_val:{max_val}, min_val:{min_val}")
                if max_val > min_val:  # 避免除零
                    np_image = (np_image - min_val) / (max_val - min_val)
                    # save_path = "./convert_depth_images"
                    # os.makedirs(save_path, exist_ok=True)
                    # save_color_map(np_image, os.path.join(save_path, f"normalized_0.png"))
                else:
                    raise ValueError("Invalid depth range. Max == Min.")
            else:
                raise ValueError(f"Unsupported image mode: {image.mode}")
            
            # # 保存归一化后的伪彩色图像
            # # 生成时间戳
            # timestamp = time.strftime("%Y%m%d-%H%M%S")
            # save_color_map(np_image, os.path.join(save_path, f"normalized_{timestamp}.png"))

            # 转换为 PyTorch 张量并添加通道维度
            torch_image = torch.from_numpy(np_image).unsqueeze(0)  # Shape: 1 x H x W
            torch_images.append(torch_image)

        return torch.stack(torch_images)  # Shape: B x 1 x H x W

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for i, root in enumerate(self.cfg.roots):
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()), self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.train_times_per_scene
        )

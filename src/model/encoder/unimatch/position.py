# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py

import torch
import torch.nn as nn
import math


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    num_pos_feats: 每个空间维度（高度和宽度）的位置编码特征数（默认为64）
    temperature: 正弦和余弦的周期因子，控制正弦波的频率（默认为10000）
    normalize: 是否对位置索引引进行归一化。若为True，位置索引值会被归一化到[0，scale]范围。
    scale: 若启用归一化（normalize=True），用于定义归一化后的位置编码范围（默认为2 * pi）
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        # 构造掩码：生成一个全 1 的掩码张量（mask），形状为 [B, H, W]
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        # 对掩码的高度维度（1）和宽度维度（2）分别计算累加和，生成位置索引
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        # 将位置索引归一化到 [0, scale] 范围
        if self.normalize:
            eps = 1e-6
            # 除以每列的最大值 再乘以scale
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # dim_t 是一个 1D 张量，表示不同通道的频率因子
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 将 x_embed 和 y_embed 位置索引分别除以频率因子 dim_t，形状变为 [B, H, W, num_pos_feats]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 0::2 和 1::2 分别提取偶数和奇数维度，分别计算 sin 和 cos
        # 使用 torch.stack 合并后，再通过 flatten 压缩到形状 [B, H, W, num_pos_feats*2]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        # 将垂直位置编码 pos_y 和水平位置编码 pos_x 进行拼接，得到形状 [B, H, W, num_pos_feats*2]
        # 使用 permute 调整维度顺序为 [B, num_pos_feats*2, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

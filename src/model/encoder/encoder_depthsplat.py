from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch, set_num_views
from .unimatch.ldm_unet.unet import UNetModel
from .unimatch.feature_upsampler import ResizeConvFeatureUpsampler

# unet3d
from .unimatch.unet3d.model import get_model

from colorama import Fore
def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int
    no_mapping: bool


@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerDepthSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    costvolume_nearest_n_views: Optional[int] = None
    multiview_trans_nearest_n_views: Optional[int] = None


class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)

        # 深度估计的对象
        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        # upsample to the original resolution
        self.feature_upsampler = ResizeConvFeatureUpsampler(num_scales=cfg.num_scales,
                                                            lowest_feature_resolution=cfg.lowest_feature_resolution,
                                                            out_channels=self.cfg.feature_upsampler_channels,
                                                            vit_type=self.cfg.monodepth_vit_type,
                                                            )
        feature_upsampler_channels = self.cfg.feature_upsampler_channels

        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # unet
        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        modules = [
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        ]

        if self.cfg.color_large_unet or self.cfg.gaussian_regressor_channels == 16:
            unet_channel_mult = [1, 2, 4, 4, 4]
        else:
            unet_channel_mult = [1, 1, 1, 1, 1]
        unet_attn_resolutions = [16]

        modules.append(
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=unet_attn_resolutions,
                channel_mult=unet_channel_mult,
                num_head_channels=32 if self.cfg.gaussian_regressor_channels >= 32 else 16,
                dims=2,
                postnorm=False,
                num_frames=2,
                use_cross_view_self_attn=True,
            )
        )

        modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

        # gaussian_regressor中是unet网络结构
        self.gaussian_regressor = nn.Sequential(*modules)

        # predict gaussian parameters: scale, q, sh
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2

        # predict opacity
        num_gaussian_parameters += 1

        # concat(img, features, unet_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1

        if self.cfg.feature_upsampler_channels != 128:
            self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                            3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                            num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )
        else:
            self.gaussian_head = nn.Sequential(
                nn.Conv2d(
                    in_channels, num_gaussian_parameters * 2, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters * 2,
                            num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # init scale
        # first 3: opacity, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

    # 深度估计可能是在这进行的
    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        # debug
        # print("context[\"image\"].shape: {}".format(context["image"].shape))
        # torch.Size([1, 2, 3, 176, 320])

        b1, v1, _, h1, w1 = context["depth_image"].shape
        # debug
        # context["depth_image"].shape: torch.Size([1, 2, 1, 176, 320])
        # print("context[\"depth_image\"].shape: {}".format(context["depth_image"].shape))

        if (
            self.cfg.costvolume_nearest_n_views is not None
            or self.cfg.multiview_trans_nearest_n_views is not None
        ):
            assert self.cfg.costvolume_nearest_n_views is not None
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:,
                                                        :, :self.cfg.costvolume_nearest_n_views]
        else:
            cameras_dist_index = None

        # depth prediction
        # MultiViewUniMatch类中的forward函数
        results_dict = self.depth_predictor(
            context["image"],
            context["depth_image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )

        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']
        # debug
        # print(f"type(depth_preds): {type(depth_preds)})")
        # for i in range(len(depth_preds)):
        #     print(f"depth_preds[{i}].shape: {depth_preds[i].shape}")
        # for i in range(len(context["depth_image"])):
        #     print(f"context[\"depth_image\"][{i}].shape: {context['depth_image'][i].shape}")

        # 使用真实的深度图
        # use_true_depth = False
        # if use_true_depth:
        #     depth_preds[-1][:] = context["depth_image"].squeeze(2)

        # [B, V, H, W]
        depth = depth_preds[-1]
        # debug type(depth): <class 'torch.Tensor'>  depth.shape: torch.Size([1, 2, 176, 320])
        # print(f"-----------------depth real----------------\n: {depth_preds[-1]}")

        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }

        # update the num_views in unet attention, useful for random input views
        set_num_views(self.gaussian_regressor, v)

        # features [BV, C, H, W]
        features = self.feature_upsampler(results_dict["features_cnn"],
                                            results_dict["features_mv"],
                                            results_dict["features_mono"],
                                            )

        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')

        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)

        # 把concat作为unet的输入
        out = self.gaussian_regressor(concat)
        # debug concat.shape: torch.Size([2, 69, 176, 320])
        # debug out.shape: torch.Size([2, 16, 176, 320])
        # print(f"[encoder_depthsplat] out.shape: {out.shape}")

        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    match_prob]

        out = torch.cat(concat, dim=1)
        # debug out_1.shape: torch.Size([2, 84, 176, 320])
        # print(f"[encoder_depthsplat] out_1.shape: {out.shape}")

        gaussians = self.gaussian_head(out)  # [BV, C, H, W]
        # debug gaussians.shape: torch.Size([2, 85, 176, 320])
        # print(f"[encoder_depthsplat] gaussians.shape: {gaussians.shape}")

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)
        # debug [encoder_depthsplat] gaussians_1.shape: torch.Size([1, 2, 85, 176, 320])
        # print(f"[encoder_depthsplat] gaussians_1.shape: {gaussians.shape}")

        depths = rearrange(depth, "b v h w -> b v (h w) () ()")

        # [B, V, H*W, 1, 1]
        densities = rearrange(
            match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(
            gaussians, "b v c h w -> b v (h w) c")

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

            # supervise all the intermediate depth predictions
            num_depths = len(depth_preds)

            # [B, V, H*W, 1, 1]
            intermediate_depths = torch.cat(
                depth_preds[:(num_depths - 1)], dim=0)

            intermediate_depths = rearrange(
                intermediate_depths, "b v h w -> b v (h w) () ()")

            # concat in the batch dim
            depths = torch.cat((intermediate_depths, depths), dim=0)

            # shared color head
            densities = torch.cat([densities] * num_depths, dim=0)
            raw_gaussians = torch.cat(
                [raw_gaussians] * num_depths, dim=0)

            b *= num_depths
            
        # debug raw_gaussians.shape: torch.Size([2, 2, 56320, 85])
        # print(f"[encoder_depthsplat] raw_gaussians.shape: {raw_gaussians.shape}")

        # [B, V, H*W, 1, 1]
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
        raw_gaussians = raw_gaussians[..., 1:]

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        sh_input_images = context["image"]

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            context_extrinsics = torch.cat(
                [context["extrinsics"]] * len(depth_preds), dim=0)
            context_intrinsics = torch.cat(
                [context["intrinsics"]] * len(depth_preds), dim=0)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images.repeat(
                    len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
            )

        else:
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images if self.cfg.init_sh_input_img else None,
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        if self.cfg.return_depth:
            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": gaussians,
                "depths": depths
            }

        return gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None

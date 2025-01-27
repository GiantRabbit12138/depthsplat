#!/bin/bash

# evaluate on cuboid
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=cuboid \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=1 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_regressor_channels=16 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=64 \
mode=test \
test.compute_scores=true \
wandb.mode=disabled \
test.save_image=true \
test.save_gt_image=true \
checkpointing.pretrained_model=/root/autodl-tmp/cuboid-image4_with_depth_outputs_20250126/checkpoints/epoch_9999-step_10000.ckpt \
output_dir=outputs/tmp_with_depth_3d_unet_20250127
# dataset/view_sampler=all
# model.encoder.monodepth_vit_type=vitb 加了这个就会报错

# render video on cuboid (need to have ffmpeg installed)
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=cuboid \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=1 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_regressor_channels=16 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=64 \
checkpointing.pretrained_model=/root/autodl-tmp/cuboid-image4_with_depth_outputs_20250126/checkpoints/epoch_9999-step_10000.ckpt \
mode=test \
test.save_video=true \
test.compute_scores=false \
wandb.mode=disabled \
test.save_image=false \
test.save_gt_image=false \
output_dir=outputs/tmp_with_depth_3d_unet_20250127
# dataset/view_sampler=all
# model.encoder.monodepth_vit_type=vitb \
# dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \

#!/bin/bash

# evaluate on cuboid
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=cup_cuboid_2 \
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
checkpointing.pretrained_model=/root/autodl-tmp/cup_cuboid_720_1280_outputs_20250215/checkpoints/epoch_9999-step_10000.ckpt \
output_dir=outputs/cup_cuboid_720_1280_eval
# dataset/view_sampler=all
# model.encoder.monodepth_vit_type=vitb 加了这个就会报错

# render video on cuboid (need to have ffmpeg installed)
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=cup_cuboid_2 \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=1 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_regressor_channels=16 \
model.encoder.color_large_unet=true \
model.encoder.feature_upsampler_channels=64 \
checkpointing.pretrained_model=/root/autodl-tmp/cup_cuboid_720_1280_outputs_20250215/checkpoints/epoch_9999-step_10000.ckpt \
mode=test \
test.save_video=true \
test.compute_scores=false \
wandb.mode=disabled \
test.save_image=false \
test.save_gt_image=false \
output_dir=outputs/cup_cuboid_720_1280_eval
# dataset/view_sampler=all
# model.encoder.monodepth_vit_type=vitb \
# dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \

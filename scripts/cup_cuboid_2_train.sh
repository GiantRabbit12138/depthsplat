#!/bin/bash

# export HYDRA_FULL_ERROR=1

python -m src.main +experiment=cup_cuboid_2 \
data_loader.train.batch_size=2 \
dataset.test_chunk_interval=10 \
trainer.val_check_interval=0.5 \
trainer.max_steps=10000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_regressor_channels=16 \
model.encoder.feature_upsampler_channels=64 \
model.encoder.return_depth=true \
wandb.project=depthsplat \
output_dir=/root/autodl-tmp/cup_cuboid_720_1280_outputs_20250215 \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \
dataset.near=0.5 \
dataset.far=100.0

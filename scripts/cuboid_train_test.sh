#!/bin/bash

# export HYDRA_FULL_ERROR=1

python -m src.main +experiment=cuboid \
data_loader.train.batch_size=1 \
dataset.test_chunk_interval=10 \
trainer.val_check_interval=0.5 \
trainer.max_steps=10000 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.gaussian_regressor_channels=16 \
model.encoder.feature_upsampler_channels=64 \
model.encoder.return_depth=true \
wandb.project=depthsplat \
output_dir=/root/autodl-tmp/cuboid-image4_with_depth_outputs_20250126 \
checkpointing.pretrained_monodepth=pretrained/depth_anything_v2_vits.pth \
checkpointing.pretrained_mvdepth=pretrained/gmflow-scale1-things-e9887eda.pth \

<p align="center">
  <h1 align="center">DepthSplat: Connecting Gaussian Splatting and Depth</h1>
  <p align="center">
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    ·
    <a href="https://pengsongyou.github.io/">Songyou Peng</a>
    ·
    <a href="https://fangjinhuawang.github.io/">Fangjinhua Wang</a>
    ·
    <a href="https://hermannblum.net/">Hermann Blum</a>
    ·
    <a href="https://scholar.google.com/citations?user=U9-D8DYAAAAJ">Daniel Barath</a>
    ·
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>
    ·
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2410.13862">Paper</a> | <a href="https://haofeixu.github.io/depthsplat/">Project Page</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://haofeixu.github.io/depthsplat/assets/teaser.png" alt="Logo" width="100%">
  </a>
</p>


<p align="center">
<strong>DepthSplat enables cross-task interactions between Gaussian splatting and depth estimation.</strong>
</p>

## Directory Structure

```bash
depthsplat
├── DATASETS.md
├── LICENSE
├── MODEL_ZOO.md
├── README.md
├── assets
│   ├── dl3dv_start_0_distance_100_ctx_12v_tgt_16v_video.json
│   ├── dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json
│   ├── dl3dv_start_0_distance_50_ctx_2v_tgt_4v_video_0-50.json
│   ├── dl3dv_start_0_distance_50_ctx_4v_tgt_4v_video_0-50.json
│   ├── dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json
│   ├── dl3dv_start_0_distance_50_ctx_6v_tgt_8v_video_0-50.json
│   ├── evaluation_index_re10k.json
│   └── evaluation_index_re10k_video.json
├── config
│   ├── dataset
│   │   ├── dl3dv.yaml
│   │   ├── re10k.yaml
│   │   ├── view_sampler
│   │   └── view_sampler_dataset_specific_config
│   ├── experiment
│   │   ├── dl3dv.yaml
│   │   └── re10k.yaml
│   ├── loss
│   │   ├── lpips.yaml
│   │   └── mse.yaml
│   ├── main.yaml
│   └── model
│       ├── decoder
│       └── encoder
├── datasets
│   └── re10k -> /root/autodl-tmp/re10k
├── download_dataset.py
├── pretrained
│   ├── depthsplat-gs-base-re10k-256x256-044fdb17.pth -> /root/autodl-tmp/models/depthsplat-gs-base-re10k-256x256-044fdb17.pth
│   ├── depthsplat-gs-large-re10k-256x256-288d9b26.pth -> /root/autodl-tmp/models/depthsplat-gs-large-re10k-256x256-288d9b26.pth
│   └── depthsplat-gs-small-re10k-256x256-49b2d15c.pth -> /root/autodl-tmp/models/depthsplat-gs-small-re10k-256x256-49b2d15c.pth
├── requirements.txt
├── scripts
│   ├── dl3dv_256x448_depthsplat_base.sh
│   ├── inference_depth_base.sh
│   ├── inference_depth_large.sh
│   ├── inference_depth_small.sh
│   ├── re10k_256x256_depthsplat_base.sh
│   ├── re10k_256x256_depthsplat_large.sh
│   └── re10k_256x256_depthsplat_small.sh
└── src
    ├── __pycache__
    │   ├── config.cpython-310.opt-jaxtyping983a4111806314cc973c4ea00fb072bf6.pyc
    │   ├── global_cfg.cpython-310.opt-jaxtyping983a4111806314cc973c4ea00fb072bf6.pyc
    │   └── main.cpython-310.pyc
    ├── config.py
    ├── dataset
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── data_module.py
    │   ├── dataset.py
    │   ├── dataset_dl3dv.py
    │   ├── dataset_re10k.py
    │   ├── shims
    │   ├── types.py
    │   ├── validation_wrapper.py
    │   └── view_sampler
    ├── evaluation
    │   ├── __pycache__
    │   ├── evaluation_cfg.py
    │   ├── evaluation_index_generator.py
    │   ├── metric_computer.py
    │   └── metrics.py
    ├── geometry
    │   ├── __pycache__
    │   ├── epipolar_lines.py
    │   └── projection.py
    ├── global_cfg.py
    ├── loss
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── loss.py
    │   ├── loss_lpips.py
    │   └── loss_mse.py
    ├── main.py
    ├── misc
    │   ├── LocalLogger.py
    │   ├── __pycache__
    │   ├── benchmarker.py
    │   ├── collation.py
    │   ├── discrete_probability_distribution.py
    │   ├── heterogeneous_pairings.py
    │   ├── image_io.py
    │   ├── nn_module_tools.py
    │   ├── render_utils.py
    │   ├── resume_ckpt.py
    │   ├── sh_rotation.py
    │   ├── stablize_camera.py
    │   ├── step_tracker.py
    │   └── wandb_tools.py
    ├── model
    │   ├── __pycache__
    │   ├── decoder
    │   ├── encoder
    │   ├── model_wrapper.py
    │   ├── ply_export.py
    │   └── types.py
    ├── scripts
    │   ├── convert_dl3dv_test.py
    │   ├── convert_dl3dv_train.py
    │   └── generate_dl3dv_index.py
    └── visualization
        ├── __pycache__
        ├── annotation.py
        ├── camera_trajectory
        ├── color_map.py
        ├── colors.py
        ├── drawing
        ├── layout.py
        ├── validation_in_3d.py
        └── vis_depth.py
```



## Installation

Our code is developed based on pytorch 2.4.0, CUDA 12.4 and python 3.10. 

We recommend using [conda](https://docs.anaconda.com/miniconda/) for installation:

```bash
conda create -y -n depthsplat python=3.10
conda activate depthsplat

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install xformers==0.0.27.post2
pip install -r requirements.txt
```

## Model Zoo

Our models are hosted on Hugging Face 🤗 : https://huggingface.co/haofeixu/depthsplat

Model details can be found at [MODEL_ZOO.md](MODEL_ZOO.md).

We assume the downloaded weights are located in the `pretrained` directory.

## Camera Conventions

The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height).

The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).

## Datasets

Please refer to [DATASETS.md](DATASETS.md) for dataset preparation.



## Depth Prediction

Please check [scripts/inference_depth_small.sh](scripts/inference_depth_small.sh), [scripts/inference_depth_base.sh](scripts/inference_depth_base.sh), and [scripts/inference_depth_large.sh](scripts/inference_depth_large.sh) for scale-consistent depth prediction with models of different sizes.

![depth](https://haofeixu.github.io/depthsplat/assets/depth/c37109a55effe0000f8e40652ca935376e75bcb2a0b56de8eabd20a26e2a0f68.png)



We plan to release a simple depth inference pipeline in [UniMatch repo](https://github.com/autonomousvision/unimatch).



## Gaussian Splatting

- The training, evaluation, and rendering scripts on RealEstate10K dataset are available at [scripts/re10k_256x256_depthsplat_small.sh](scripts/re10k_256x256_depthsplat_small.sh), [scripts/re10k_256x256_depthsplat_base.sh](scripts/re10k_256x256_depthsplat_base.sh), and [scripts/re10k_256x256_depthsplat_large.sh](scripts/re10k_256x256_depthsplat_large.sh).

- The training, evaluation, and rendering scripts on DL3DV dataset are available at [scripts/dl3dv_256x448_depthsplat_base.sh](scripts/dl3dv_256x448_depthsplat_base.sh).

- Before training, you need to download the pre-trained [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [UniMatch](https://github.com/autonomousvision/unimatch) weights and set up your [wandb account](config/main.yaml) for logging.

```
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -P pretrained
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth -P pretrained
```


## Citation

```
@article{xu2024depthsplat,
      title   = {DepthSplat: Connecting Gaussian Splatting and Depth},
      author  = {Xu, Haofei and Peng, Songyou and Wang, Fangjinhua and Blum, Hermann and Barath, Daniel and Geiger, Andreas and Pollefeys, Marc},
      journal = {arXiv preprint arXiv:2410.13862},
      year    = {2024}
    }
```



## Acknowledgements

This project is developed with several fantastic repos: [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [DL3DV](https://github.com/DL3DV-10K/Dataset). We thank the original authors for their excellent work.


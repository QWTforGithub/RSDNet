# RSDNet

This repo is the official project repository of the paper **_Robust Single-Stage Fully Sparse 3D Object Detection via Detachable Latent Diffusion_**. 
 -  [ [arXiv](https://arxiv.org/pdf/2508.03252) ]
 -  The code will be released shortly...
## The Overall Framework 
...
## Citation
If you find our paper useful to your research, please cite our work as an acknowledgment.
```bib
@article{qu2025robust,
  title={Robust Single-Stage Fully Sparse 3D Object Detection via Detachable Latent Diffusion},
  author={Qu, Wentao and Mei, Guofeng and Wang, Jing and Wu, Yujiao and Huang, Xiaoshui and Xiao, Liang},
  journal={arXiv preprint arXiv:2508.03252},
  year={2025}
}
```

## Motivation
...


## Overview
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Zoo](#model-zoo)
- [Quick Start](#quick-start)

## Installation

### Requirements
The following environment is recommended for running **_RSDNet_** (an NVIDIA 3090 GPU or four NVIDIA 4090 GPUs):
- Ubuntu: 18.04 and above
- gcc/g++: 7.5 and above
- CUDA: 11.6 and above
- PyTorch: 1.13.1 and above
- python: 3.8 and above

### Environment

- Base environment
```
conda create -n dlf python=3.8 -y
conda activate dlf
conda install ninja -y

```

## Data Preparation
- ...

### nuScenes
- Download the official [nuScenes](https://www.nuscenes.org/nuscenes#download) (or [Baidu Disk](https://pan.baidu.com/s/1Rsbi-Q_2EUm05lwQgn8T3Q?pwd=1111)(code:1111)) dataset (with Lidar Segmentation) and organize the downloaded files as follows:
  ```bash
  NUSCENES_DIR
  │── samples
  │── sweeps
  │── lidarseg
  ...
  │── v1.0-trainval 
  │── v1.0-test
  ```
- The preprocess nuScenes information data can also be downloaded [[here](https://huggingface.co/datasets/Pointcept/nuscenes-compressed)] (only processed information, still need to download raw dataset and link to the folder), please agree the official license before download it.

- Link raw dataset to processed NuScene dataset folder:
  ```bash
  # NUSCENES_DIR: the directory of downloaded nuScenes dataset.
  # PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
  ln -s ${NUSCENES_DIR} {PROCESSED_NUSCENES_DIR}/raw
  ```
  then the processed nuscenes folder is organized as follows:
  ```bash
  nuscene
  |── raw
      │── samples
      │── sweeps
      │── lidarseg
      ...
      │── v1.0-trainval
      │── v1.0-test
  |── info
  ```

- Link processed dataset to codebase.
  ```bash
  # PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
  mkdir data
  ln -s ${PROCESSED_NUSCENES_DIR} ${CODEBASE_DIR}/data/nuscenes
  ```

## Model Zoo
| Model | Benchmark | Only Training Data? | Num GPUs | Val NDS | Val mAP | checkpoint |
| :---: | :---: |:---------------:| :---: | :---: | :---: | :---: |
| RSDNet | nuScenes |     &check;     | 8 | 71.9 | 69.3% | [Link1](https://pan.baidu.com/s/1rAUHa4OHmT_Q1I2Pi_sVog?pwd=1111), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |


## Quick Start

### Training
...


### Testing
...

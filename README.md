# Heterogeneous Network Based Edge-Cloud Cooperative Intelligence for Video Recognition

## Introduction
This is the code repository for our paper "Heterogeneous Network Based Edge-Cloud Cooperative Intelligence for Video Recognition". The code in this repository is based on the [MMAction2](https://github.com/open-mmlab/mmaction2) framework. Please refer to the official documentation for more information.

## Quick Start

```bash
git clone https://github.com/22222222z/Edge-Cloud-Video-Recognition.git
cd Edge-Cloud-Video-Recognition
bash scripts/quick_start.sh
```

## Training
```bash
bash scripts/run_train.sh
```

## Testing
```bash
bash scripts/run_test.sh
```

## Configuration Files
All configuration files are located in the `configs/` directory. These files define various settings and parameters for training, testing, and model setup. Here are some key configuration files:

- `configs/recognition/tsm/edge_model_k400.py`: The config file for training the edge model on Kinetics-400 dataset.
- `configs/recognition/tsm/edge_model_k400_distill.py`: The config file for distillation training on Kinetics-400 dataset.
- `configs/recognition/uniformerv2-cloud_model/uniformerv2_ECNet_K400.py`: The configuration file used for the cooperative training within our paper.
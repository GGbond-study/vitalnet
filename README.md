# ViTaL: A Multimodality Dataset and Benchmark for Multi-pathological Ovarian Tumor Recognition
This repo is the implementation of "ViTaL: A Multimodality Dataset and Benchmark for Multi-pathological Ovarian Tumor Recognition".

This repository is built on top of mmpretrain (a powerful open-source toolbox for image classification and retrieval). To fully utilize the functionalities provided in this repo, you are required to master the basic usage of mmpretrain first. Below is the essential guide for installation and model training.
1. Prerequisite: Install mmpretrain from Source
Since this project relies on the latest features of mmpretrain, we strongly recommend installing mmpretrain from its official source code instead of using pip. Follow these steps:
Step 1: Install PyTorch and TorchVision
First, ensure you have PyTorch and TorchVision installed. Refer to the official PyTorch website for commands compatible with your OS and CUDA version. For example (CUDA 11.8):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Step 2: Install MMCV (MMCV is a core dependency of mmpretrain)
Install MMCV using MIM (a package manager for OpenMMLab projects):
```bash
pip install openmim
mim install mmcv-full
```

Step 3: Clone mmpretrain and Install from Source
```bash
# Clone the mmpretrain repository
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain

# Install mmpretrain in editable mode
pip install -e .
```
Step 4: Verify Installation
Run the following command to confirm mmpretrain is installed successfully:
```bash
python -c "from mmpretrain import __version__; print(__version__)"
```
2. Train Models for Different Modalities
After installing mmpretrain and setting up this repository, use the following commands to start training for different recognition tasks. All training configurations are stored in the ./configs/ directory.
2.1 Image Single-Modal Recognition
To train the model for image-only single-modal recognition, run:
```bash
python tools/train.py ./configs/moblie_6.py
```

2.2 Image-Table-Text Multi-Modal Recognition
To train the model for image-table-text multi-modal recognition (fusing image, table, and text data), run:
```bash
python tools/train.py ./configs/vtt_moblie_6.py
```

Refer to the mmpretrain Official Documentation for more details on config file settings, training tricks, and model evaluation.


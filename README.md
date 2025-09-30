# STAR: A Benchmark for Astronomical Star Fields Super-Resolution

This is the official codebase for the paper:  
**STAR: A Benchmark for Astronomical Star Fields Super-Resolution**  
[ArXiv](https://arxiv.org/abs/2507.16385) | [Hugging Face Dataset](https://huggingface.co/datasets/KUOCHENG/STAR)

<img src="overview/icon_.png" alt="Dataset Overview" style="zoom:35%;" />

## License

This project is licensed under the [MIT License](LICENSE.md).

## Highlight

- *Sep 19, 2025*: **STAR** is selected as one of the 56 **Spotlight** by NeurIPS'25 Datasets and Benchmark Tracks! 🎉🎉🎉
- *Jul 22, 2025*: **STAR** has been released.

## Introduction

The **STAR (Super-Resolution for Astronomical Star Fields)** dataset is a large-scale benchmark for developing field-level super-resolution models in astronomy. It contains **54,738 flux-consistent image pairs** derived from Hubble Space Telescope (HST) high-resolution observations and physically faithful low-resolution counterparts. The dataset addresses three key challenges in astronomical super-resolution:

- **Flux Inconsistency**: Ensures consistent flux using a flux-preserving data generation pipeline.
- **Object-Crop Configuration**: Strategically samples patches across diverse celestial regions.
- **Data Diversity**: Covers dense star clusters, sparse galactic fields, and regions with varying background noise.

The dataset includes x2 and x4 scaling pairs in `.npy` format, suitable for training and evaluating super-resolution models.

![Sample Image Pair](overview/picture.jpg)

## Dataset Structure and Download

- **Full Data** (Hugging Face):
  - **x2 Dataset**: `data/x2/x2.tar.gz`
    - Folders: `train_hr_patch/` (HR training patches), `train_lr_patch/` (LR training patches), `eval_hr_patch/` (HR validation patches), `eval_lr_patch/` (LR validation patches), `dataload_filename/` (`.txt` files with HR/LR pair mappings).
  - **x4 Dataset**: `data/x4/x4.tar.gz`
    - Same structure as x2, for x4 scaling.
  - Download: [Hugging Face - KUOCHENG/STAR](https://huggingface.co/datasets/KUOCHENG/STAR)

- **Sample Data** (Hugging Face, for testing):
  - **x2 Sample**: `sampled_data/x2/`
    - Contains 500 HR/LR pairs in `train_hr_patch/` and `train_lr_patch/`, 100 pairs in `eval_hr_patch/` and `eval_lr_patch/` (total ~1200 `.npy` files).
  
   Quick Start:
   ```python
   from datasets import load_dataset
  import numpy as np
  dataset = load_dataset("KUOCHENG/STAR")
  sample = dataset['train'][0]
  hr_path = sample['hr_path']  # Path to HR .npy file
  lr_path = sample['lr_path']  # Path to LR .npy file
  
  hr_data = np.load(hr_path, allow_pickle=True).item()
  lr_data = np.load(lr_path, allow_pickle=True).item()
   ```
- **Source Data** (Optional):
  - Raw HST images used to generate patches.
  - Download: [Google Drive](https://drive.google.com/file/d/1SoNXzfoeY5x-mLJrMGv2pbrgm9bULzDU/view?usp=drive_link)
  
   We will update the content later (how to use this source data)

## Download

### 1. Full Dataset

Download the complete datasets for x2 and x4 scaling from Hugging Face:

- **x2**: `data/x2/x2.tar.gz` 
- **x4**: `data/x4/x4.tar.gz` 

**Usage**:

1. Download the `.tar.gz` file(s).

2. Extract to your project directory under `dataset/`:

   ```bash
   tar -xzf x2.tar.gz -C dataset/  # or x4.tar.gz
   ```

   The extracted structure will be (x2 for example):
   ```
   dataset/x2/
   ├── train_hr_patch/
   ├── train_lr_patch/
   ├── eval_hr_patch/
   ├── eval_lr_patch/
   ├── dataload_filename/
   │   ├── train_dataloader.txt
   │   ├── eval_dataloader.txt
   
   ```
## Getting Start

### 1. Environment config

We recommend using Conda for environment management.

```bash
# 1. Create and activate the conda environment
conda create -n star python=3.10 -y
conda activate star

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare the Dataset
# Download the dataset from Hugging Face (or your source)
# and unzip it into the `dataset/` directory.
```

### 2. Training

All model configurations are located in the `configs/models` directory.

#### Single-GPU Training

To train on a single GPU (e.g., GPU 0):

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/dist_trainval.sh configs/models/FISR.py --log_dir log/
```

#### Multi-GPU Training

To train on multiple GPUs (e.g., GPUs 0, 1, 2, 3):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_trainval.sh configs/models/FISR.py --log_dir log/
```

#### Custom Training Options

You can append arguments to the training command to customize the run.

- `--use_loss [L1/L2]`: Specify the loss function to use (`L1` or `L2`).
- `--use_attention`: Enable the flux loss

### 3. Testing & Evaluation

Evaluate your model using a saved checkpoint.

- `-e` or `--evaluate`: Switches the script to evaluation mode.
- `--resume 'checkpoint_path'`: Specifies the path to the checkpoint file to load.
- `-v` or `--visualize`: (Optional) Enables visualization output during testing. The output path can be configured in the model's config file.

#### Example:

```bash
# Run evaluation and save visualization results
CUDA_VISIBLE_DEVICES=0 bash tools/dist_trainval.sh configs/models/FISR.py \
    -e \
    --resume 'path/to/your/checkpoint.pth' \
    -v
```

## Citation

If you use the STAR Dataset, please cite:
```bibtex
@article{wu2025star,
  title={STAR: A Benchmark for Astronomical Star Fields Super-Resolution},
  author={Wu, Kuo-Cheng and Zhuang, Guohang and Huang, Jinyang and Zhang, Xiang and Ouyang, Wanli and Lu, Yan},
  journal={arXiv preprint arXiv:2507.16385},
  year={2025},
  url={https://arxiv.org/abs/2507.16385}
}
```

## Contact

For issues or questions, open a GitHub issue or send me an email [12guocheng@gmail.com] for free.

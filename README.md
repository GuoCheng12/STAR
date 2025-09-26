# STAR: A Benchmark for Astronomical Star Fields Super-Resolution

This is the official codebase for the paper:  
**STAR: A Benchmark for Astronomical Star Fields Super-Resolution**  
[ArXiv](https://arxiv.org/abs/2507.16385) | [Hugging Face Dataset](https://huggingface.co/datasets/KUOCHENG/STAR)

![Dataset Overview](overview/icon.png)

## License

This project is licensed under the [MIT License](LICENSE.md).

## Introduction

The **STAR (Super-Resolution for Astronomical Star Fields)** dataset is a large-scale benchmark for developing field-level super-resolution models in astronomy. It contains **54,738 flux-consistent image pairs** derived from Hubble Space Telescope (HST) high-resolution observations and physically faithful low-resolution counterparts. The dataset addresses three key challenges in astronomical super-resolution:

- **Flux Inconsistency**: Ensures consistent flux using a flux-preserving data generation pipeline.
- **Object-Crop Configuration**: Strategically samples patches across diverse celestial regions.
- **Data Diversity**: Covers dense star clusters, sparse galactic fields, and regions with varying background noise.

The dataset includes x2 and x4 scaling pairs in `.npy` format, suitable for training and evaluating super-resolution models.

![Sample Image Pair](overview/picture.jpg)

## Dataset Structure

- **Full Data** (Hugging Face):
  - **x2 Dataset**: `data/x2/x2.tar.gz` (33GB)
    - Folders: `train_hr_patch/` (HR training patches), `train_lr_patch/` (LR training patches), `eval_hr_patch/` (HR validation patches), `eval_lr_patch/` (LR validation patches), `dataload_filename/` (`.txt` files with HR/LR pair mappings).
  - **x4 Dataset**: `data/x4/x4.tar.gz` (29GB)
    - Same structure as x2, for x4 scaling.
  - Download: [Hugging Face - KUOCHENG/STAR](https://huggingface.co/datasets/KUOCHENG/STAR)

- **Sample Data** (Hugging Face, for testing and Croissant compatibility):
  - **x2 Sample**: `sampled_data/x2/` (~2.4GB)
    - Contains 500 HR/LR pairs in `train_hr_patch/` and `train_lr_patch/`, 100 pairs in `eval_hr_patch/` and `eval_lr_patch/` (total ~1200 `.npy` files).
    - Croissant metadata: `sampled_data/x2/croissant.json`
  - Download: [Hugging Face - KUOCHENG/STAR](https://huggingface.co/datasets/KUOCHENG/STAR)

- **Source Data** (Optional):
  - Raw HST images used to generate patches.
  - Download: [Google Drive](https://drive.google.com/file/d/1SoNXzfoeY5x-mLJrMGv2pbrgm9bULzDU/view?usp=drive_link)

## Download and Usage

### 1. Full Dataset

Download the complete datasets for x2 and x4 scaling from Hugging Face:

- **x2**: `data/x2/x2.tar.gz` (33GB)
- **x4**: `data/x4/x4.tar.gz` (29GB)

**Usage**:

1. Download the `.tar.gz` file(s).

2. Extract to your project directory under `dataset/`:

   ```bash
   tar -xzf x2.tar.gz -C dataset/  # or x4.tar.gz
   ```

   The extracted structure will be:

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

### 2. Sample Dataset

Use the sample data for quick testing or model prototyping:

```python
from datasets import Dataset
import numpy as np
import glob

# Load sample data
dataset = load_dataset("KUOCHENG/STAR", data_dir="sampled_data/x2/", split="train")

# Custom paired loading
train_hr_files = sorted(glob.glob("sampled_data/x2/train_hr_patch/*.npy"))
train_lr_files = [f.replace("hr_patch", "lr_patch") for f in train_hr_files]
eval_hr_files = sorted(glob.glob("sampled_data/x2/eval_hr_patch/*.npy"))
eval_lr_files = [f.replace("hr_patch", "lr_patch") for f in eval_hr_files]
data_dict = {
    "hr_data": [np.load(f) for f in train_hr_files + eval_hr_files],
    "lr_data": [np.load(f) for f in train_lr_files + eval_lr_files],
    "split": ["train"] * len(train_hr_files) + ["eval"] * len(eval_hr_files)
}
paired_dataset = Dataset.from_dict(data_dict)
print(paired_dataset[0]["hr_data"].shape)  # Output: (256, 256, 1)
```

### 3. Source Data

For raw HST images (pre-patched), download from:

- [Google Drive](https://drive.google.com/file/d/1SoNXzfoeY5x-mLJrMGv2pbrgm9bULzDU/view?usp=drive_link)

## Notes

- **Patch Size**: HR patches are `.npy` arrays with shape `(256, 256, 1)` (grayscale, single channel). LR patches have corresponding downscaled shapes.
- **Pair Information**: HR/LR pairs are listed in `train_dataloader.txt` (500 pairs in sample, thousands in full dataset) and `eval_dataloader.txt` (100 pairs in sample). Format: `hr_path,lr_path,coordinates` (ignore coordinates for most tasks).
- **Croissant Compatibility**: Sample includes `sampled_data/x2/croissant.json` for ML frameworks.
- **Performance**: Sample data (~2.4GB, 1200 files) is lightweight for testing; full datasets are large (33GB/29GB).

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

import os
from tqdm import tqdm
import numpy as np
from astropy.io import fits
from scipy.ndimage import generate_binary_structure, binary_erosion

def load_data(file_path, hr=True):
    """加载 FITS 文件，返回图像数据和掩码"""
    with fits.open(file_path) as hdul:
        img_data = hdul[0].data.astype(float)
        zero_mask = (img_data == 0)
        structure = generate_binary_structure(2, 1)
        eroded_zero_mask = binary_erosion(zero_mask, structure=structure)
        img_data[eroded_zero_mask] = np.nan
        mask = ~np.isnan(img_data)
        return img_data, mask

def patchify_hr(image, mask, patch_size=256, stride=128, useful_region_th=0.8):
    """HR 分块"""
    patches = []
    h, w = image.shape
    
    for row_start in range(0, h, stride):
        for col_start in range(0, w, stride):
            row_end = min(row_start + patch_size, h)
            col_end = min(col_start + patch_size, w)
            
            mask_patch = mask[row_start:row_end, col_start:col_end]
            if mask_patch.mean() < useful_region_th:
                continue
                
            patches.append((
                image[row_start:row_end, col_start:col_end],
                mask_patch,
                (row_start, col_start)
            ))
    
    return patches

def get_lr_patch(lr_image, lr_mask, hr_coord, scale_factor=2, lr_patch_size=128, useful_region_th=0.8):
    """生成 LR patch"""
    hr_row_start, hr_col_start = hr_coord
    lr_row_start = int(hr_row_start / scale_factor)
    lr_col_start = int(hr_col_start / scale_factor)
    lr_row_end = lr_row_start + lr_patch_size
    lr_col_end = lr_col_start + lr_patch_size
    lr_row_end = min(lr_row_end, lr_image.shape[0])
    lr_col_end = min(lr_col_end, lr_image.shape[1])
    
    lr_mask_patch = lr_mask[lr_row_start:lr_row_end, lr_col_start:lr_col_end]
    if lr_mask_patch.mean() < useful_region_th:
        return None
    
    return (
        lr_image[lr_row_start:lr_row_end, lr_col_start:lr_col_end],
        lr_mask_patch,
        (lr_row_start, lr_col_start)
    )

def generate_patch_pairs(hr_image, hr_mask, lr_image, lr_mask, hr_patch_size=256, lr_patch_size=128, stride=128, scale_factor=2, useful_region_th=0.8):
    """生成 HR 和 LR 的 patch 对"""
    patch_pairs = []
    hr_patches = patchify_hr(hr_image, hr_mask, hr_patch_size, stride, useful_region_th)
    
    for hr_patch, hr_mask_patch, hr_coord in hr_patches:
        lr_patch_info = get_lr_patch(lr_image, lr_mask, hr_coord, scale_factor, lr_patch_size, useful_region_th)
        if lr_patch_info:
            lr_patch, lr_mask_patch, lr_coord = lr_patch_info
            patch_pairs.append((
                hr_patch, hr_mask_patch, hr_coord,
                lr_patch, lr_mask_patch, lr_coord
            ))
    return patch_pairs

def generate_dataloader_txt(patch_pairs, hr_patch_dir, lr_patch_dir, dataloader_txt, identifier):
    """保存 HR 和 LR patch 并生成 dataloader.txt"""
    with open(dataloader_txt, "a") as f:
        for idx, (hr_patch, hr_mask, hr_coord, lr_patch, lr_mask, lr_coord) in enumerate(patch_pairs):
            hr_patch_path = os.path.join(hr_patch_dir, f"{identifier}_hr_patch_{idx}.npy")
            np.save(hr_patch_path, {"image": hr_patch, "mask": hr_mask, "coord": hr_coord})
            lr_patch_path = os.path.join(lr_patch_dir, f"{identifier}_lr_patch_{idx}.npy")
            np.save(lr_patch_path, {"image": lr_patch, "mask": lr_mask, "coord": lr_coord})
            f.write(f"{hr_patch_path},{lr_patch_path},{hr_coord}\n")

def process_patchify(train_files_path, eval_files_path, dataset_dir, dataload_filename_dir, hr_patch_size=256, lr_patch_size=128, stride=128, useful_region_th=0.8, scale_factor=2):
    """对 HR 和 LR 图像进行 Patchify 并生成训练集和验证集的 dataloader.txt"""
    train_hr_patch_dir = os.path.join(dataset_dir, "train_hr_patch")
    train_lr_patch_dir = os.path.join(dataset_dir, "train_lr_patch")
    eval_hr_patch_dir = os.path.join(dataset_dir, "eval_hr_patch")
    eval_lr_patch_dir = os.path.join(dataset_dir, "eval_lr_patch")
    os.makedirs(train_hr_patch_dir, exist_ok=True)
    os.makedirs(train_lr_patch_dir, exist_ok=True)
    os.makedirs(eval_hr_patch_dir, exist_ok=True)
    os.makedirs(eval_lr_patch_dir, exist_ok=True)
    
    os.makedirs(dataload_filename_dir, exist_ok=True)
    train_dataloader_txt = os.path.join(dataload_filename_dir, "train_dataloader.txt")
    eval_dataloader_txt = os.path.join(dataload_filename_dir, "eval_dataloader.txt")
    if os.path.exists(train_dataloader_txt):
        os.remove(train_dataloader_txt)
    if os.path.exists(eval_dataloader_txt):
        os.remove(eval_dataloader_txt)

    with open(train_files_path, "r") as f:
        train_files = [line.strip().split(',') for line in f.readlines()]

    for hr_path, lr_path in tqdm(train_files, desc="Processing train files"):
        try:
            hr_image, hr_mask = load_data(hr_path, hr=True)
            lr_image, lr_mask = load_data(lr_path, hr=False)
            identifier = os.path.basename(hr_path).replace(".fits", "").replace(".gz", "")
            print(f"processing {identifier}")
            patch_pairs = generate_patch_pairs(hr_image, hr_mask, lr_image, lr_mask,
                                               hr_patch_size=hr_patch_size, lr_patch_size=lr_patch_size, 
                                               stride=stride, scale_factor=scale_factor, useful_region_th=useful_region_th)
            if len(patch_pairs) > 0:
                generate_dataloader_txt(patch_pairs, train_hr_patch_dir, train_lr_patch_dir, train_dataloader_txt, identifier)
            else:
                print(f"warning: {hr_path} No valid patch")
        except Exception as e:
            print(f"processing {hr_path} fail: {e}")
    
    with open(eval_files_path, "r") as f:
        eval_files = [line.strip().split(',') for line in f.readlines()]

    for hr_path, lr_path in tqdm(eval_files, desc="Processing eval files"):
        try:
            hr_image, hr_mask = load_data(hr_path, hr=True)
            lr_image, lr_mask = load_data(lr_path, hr=False)
            identifier = os.path.basename(hr_path).replace(".fits", "").replace(".gz", "")
            patch_pairs = generate_patch_pairs(hr_image, hr_mask, lr_image, lr_mask,
                                               hr_patch_size=hr_patch_size, lr_patch_size=lr_patch_size, 
                                               stride=stride, scale_factor=scale_factor, useful_region_th=useful_region_th)
            if len(patch_pairs) > 0:
                generate_dataloader_txt(patch_pairs, eval_hr_patch_dir, eval_lr_patch_dir, eval_dataloader_txt, identifier)
            else:
                print(f"warning: {hr_path} No valid patch")
        except Exception as e:
            print(f"processing {hr_path} fail: {e}")

if __name__ == "__main__":
    train_files_path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_real_world/split_file/train_files.txt"
    eval_files_path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_real_world/split_file/eval_files.txt"
    dataset_dir = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_real_world/"
    dataload_filename_dir = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_real_world/dataload_filename"
    process_patchify(train_files_path, eval_files_path, dataset_dir, dataload_filename_dir)
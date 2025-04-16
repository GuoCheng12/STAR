import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from tqdm import tqdm

def load_patch_data(patch_dir):
    """加载所有 .npy 文件中的 Patch 数据"""
    patch_files = [f for f in os.listdir(patch_dir) if f.endswith('.npy')]
    patches = []
    for file in tqdm(patch_files, desc="Loading patches"):
        data = np.load(os.path.join(patch_dir, file), allow_pickle=True).item()
        patches.append(data)
    return patches

def get_original_shape(patches, patch_size):
    """根据 Patch 坐标推断原始图像的尺寸"""
    x_coords = [patch['coord'][0] for patch in patches]
    y_coords = [patch['coord'][1] for patch in patches]
    h = max(x_coords) + patch_size  # 高度
    w = max(y_coords) + patch_size  # 宽度
    return h, w

def stitch_patches(patches, h, w, patch_size):
    """将 Patch 拼接回大图，处理重叠区域"""
    big_image = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)  # 记录重叠次数
    for patch in tqdm(patches, desc="Stitching patches"):
        x_start, y_start = patch['coord']
        image_patch = patch['image']
        print(f"Patch at ({x_start}, {y_start}), shape: {image_patch.shape}, nan ratio: {np.isnan(image_patch).mean()}")
        big_image[x_start:x_start + patch_size, y_start:y_start + patch_size] += image_patch
        count_map[x_start:x_start + patch_size, y_start:y_start + patch_size] += 1
    count_map[count_map == 0] = 1
    big_image /= count_map
    return big_image

def visualize_big_image(big_image, patches):
    """可视化原始大图和 Z-scale 归一化后的结果，以及恒星位置"""
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(big_image)
    normalized_image = np.clip((big_image - vmin) / (vmax - vmin), 0, 1)
    normalized_image = np.nan_to_num(normalized_image, nan=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(30, 15))
    
    axes[0].imshow(big_image, cmap='gray', origin='lower')
    axes[0].set_title('Original Stitched HR Image with Stars')
    for patch in patches:
        x_start, y_start = patch['coord']
        for star in patch['stars']:
            global_x = x_start + star['rel_x']
            global_y = y_start + star['rel_y']
            axes[0].scatter(global_x, global_y, c='lime', s=50, marker='x')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    
    im = axes[1].imshow(normalized_image, cmap='gray', origin='lower')
    axes[1].set_title('Z-scale Normalized Stitched HR Image with Stars')
    for patch in patches:
        x_start, y_start = patch['coord']
        for star in patch['stars']:
            global_x = x_start + star['rel_x']
            global_y = y_start + star['rel_y']
            axes[1].scatter(global_x, global_y, c='lime', s=50, marker='x')
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')
    
    plt.colorbar(im, ax=axes[1], label='Normalized Intensity')
    plt.tight_layout()
    plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/data_process/meter/vis/result.png')

if __name__ == "__main__":
    hr_patch_dir = "/home/bingxing2/ailab/scxlab0061/Astro_SR/data_process/meter/dataset/train_hr_patch"
    patches = load_patch_data(hr_patch_dir)
    patch_size = 256
    stride = 128
    h, w = get_original_shape(patches, patch_size)
    print(f"推断出的原始图像尺寸: {h}x{w}")
    big_image = stitch_patches(patches, h, w, patch_size)
    visualize_big_image(big_image, patches)
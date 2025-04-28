import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from tqdm import tqdm
import random
random.seed(42)

# 设置 Z-scale 归一化函数
def z_scale_image(image):
    """对图像应用 Z-scale 归一化"""
    z = ZScaleInterval()
    vmin, vmax = z.get_limits(image)
    normalized_image = np.clip((image - vmin) / (vmax - vmin), 0, 1)  # 归一化到 [0, 1]
    return normalized_image

# 可视化 HR 和 LR patch 对，并在同一张图上标记恒星位置
def visualize_patch_pair(hr_patch, lr_patch, hr_stars, lr_stars, idx, output_dir):
    """可视化 HR 和 LR patch 对及其恒星位置，并保存到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 应用 Z-scale 归一化
    hr_patch_normalized = z_scale_image(hr_patch)
    lr_patch_normalized = z_scale_image(lr_patch)
    
    dpi = 100  # 设置 DPI，控制像素到英寸的转换
    fig = plt.figure(figsize=(hr_patch.shape[1]/dpi + lr_patch.shape[1]/dpi, max(hr_patch.shape[0], lr_patch.shape[0])/dpi), dpi=dpi)
    
    # 添加 HR patch 子图，位置和大小基于像素
    ax1 = fig.add_axes([0, 0, hr_patch.shape[1]/(hr_patch.shape[1] + lr_patch.shape[1]), 1])
    ax1.imshow(hr_patch_normalized, cmap='gray', interpolation='nearest')
    # 标记 HR patch 上的恒星位置
    for star in hr_stars:
        ax1.scatter(star['rel_x'], star['rel_y'], s=8, c='red', marker='o')
    ax1.set_title(f"HR Patch {idx} ({hr_patch.shape[0]}x{hr_patch.shape[1]})")
    ax1.axis('off')
    
    # 添加 LR patch 子图，保持原始像素大小
    ax2 = fig.add_axes([hr_patch.shape[1]/(hr_patch.shape[1] + lr_patch.shape[1]), 0, lr_patch.shape[1]/(hr_patch.shape[1] + lr_patch.shape[1]), lr_patch.shape[0]/max(hr_patch.shape[0], hr_patch.shape[0])])
    ax2.imshow(lr_patch_normalized, cmap='gray', interpolation='nearest')
    # 标记 LR patch 上的恒星位置
    for star in lr_stars:
        ax2.scatter(star['rel_x'], star['rel_y'], s=8, c='red', marker='o')
    ax2.set_title(f"LR Patch {idx} ({lr_patch.shape[0]}x{lr_patch.shape[1]})")
    ax2.axis('off')
    
    # 保存图像
    output_path = os.path.join(output_dir, f"patch_{idx}.png")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

# 主函数：加载并随机可视化 50 个 patch 对
def visualize_random_patches(hr_patch_dir, lr_patch_dir, output_dir, num_samples=50):
    """加载并随机可视化 50 个 HR 和 LR patch 对及其恒星位置"""
    # 获取 HR 和 LR patch 文件列表
    hr_files = sorted([f for f in os.listdir(hr_patch_dir) if f.endswith('.npy')])
    lr_files = sorted([f for f in os.listdir(lr_patch_dir) if f.endswith('.npy')])
    
    # 确保文件数量匹配
    if len(hr_files) != len(lr_files):
        print(f"警告: HR 文件数量 ({len(hr_files)}) 与 LR 文件数量 ({len(lr_files)}) 不一致")
        return
    
    # 生成 HR 和 LR 文件对
    patch_pairs = list(zip(hr_files, lr_files))
    
    # 随机选择 50 个样本
    if len(patch_pairs) > num_samples:
        patch_pairs = random.sample(patch_pairs, num_samples)
    else:
        print(f"样本数量不足 {num_samples}，实际数量为 {len(patch_pairs)}，将使用所有样本")
    
    # 可视化选中的样本
    for idx, (hr_file, lr_file) in enumerate(tqdm(patch_pairs, desc="Visualizing random patches")):
        # 加载 HR 和 LR patch 数据
        hr_data = np.load(os.path.join(hr_patch_dir, hr_file), allow_pickle=True).item()
        lr_data = np.load(os.path.join(lr_patch_dir, lr_file), allow_pickle=True).item()
        
        hr_patch = hr_data['image']
        lr_patch = lr_data['image']
        hr_stars = hr_data['stars']  # 提取 HR 恒星数据
        lr_stars = lr_data['stars']  # 提取 LR 恒星数据
        
        # 可视化并保存
        visualize_patch_pair(hr_patch, lr_patch, hr_stars, lr_stars, idx, output_dir)

if __name__ == "__main__":
    hr_patch_dir = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/train_hr_patch"
    lr_patch_dir = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/train_lr_patch"
    output_dir = "/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/tmp_npy"
    visualize_random_patches(hr_patch_dir, lr_patch_dir, output_dir, num_samples=50)
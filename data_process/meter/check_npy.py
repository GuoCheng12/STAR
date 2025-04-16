import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import os

def z_scale_image(image):
    """对图像应用 Z-scale 归一化"""
    z = ZScaleInterval()
    vmin, vmax = z.get_limits(image)
    normalized_image = np.clip((image - vmin) / (vmax - vmin), 0, 1)  # 归一化到 [0, 1]
    return normalized_image

def visualize_patch_pairs(hr_patch_dir, lr_patch_dir, num_pairs=5):
    """从 HR 和 LR patch 目录中提取并可视化 HR patch 中包含恒星的 patch 对，并添加不带标记的 HR patch"""
    # 获取 HR 和 LR patch 文件列表
    hr_files = sorted([f for f in os.listdir(hr_patch_dir) if f.endswith('.npy')])
    lr_files = sorted([f for f in os.listdir(lr_patch_dir) if f.endswith('.npy')])

    # 确保文件数量匹配
    if len(hr_files) != len(lr_files):
        print(f"警告: HR 文件数量 ({len(hr_files)}) 与 LR 文件数量 ({len(lr_files)}) 不一致")
        return

    # 筛选 HR patch：stars 数量 > 10 且坐标 x, y 在 [1800, 2200] 范围内
    selected_pairs = []
    for i in range(len(hr_files)):
        hr_data = np.load(os.path.join(hr_patch_dir, hr_files[i]), allow_pickle=True).item()
        hr_coord = hr_data['coord']  # 获取 HR patch 的全局坐标 (x_start, y_start)
        x_start, y_start = hr_coord
        # 检查条件：恒星数 > 10 且坐标在指定范围内
        if (len(hr_data['stars']) > 10):
            selected_pairs.append((hr_files[i], lr_files[i]))
        if len(selected_pairs) >= num_pairs:  # 达到指定数量后停止
            break

    num_selected = len(selected_pairs)
    if num_selected == 0:
        print("没有找到符合条件的 HR patch（恒星数 > 10 且坐标在 [1800, 2200] 范围内）")
        return

    # 创建子图，动态调整行数，每行 3 列
    fig, axes = plt.subplots(num_selected, 3, figsize=(20, 5 * num_selected))

    for i, (hr_file, lr_file) in enumerate(selected_pairs):
        # 加载 HR patch 数据
        hr_data = np.load(os.path.join(hr_patch_dir, hr_file), allow_pickle=True).item()
        hr_image = hr_data['image']
        hr_stars = hr_data['stars']
        hr_coord = hr_data['coord']

        # 加载 LR patch 数据
        lr_data = np.load(os.path.join(lr_patch_dir, lr_file), allow_pickle=True).item()
        lr_image = lr_data['image']
        lr_stars = lr_data['stars']
        lr_coord = lr_data['coord']

        # 应用 Z-scale 归一化
        hr_vis = z_scale_image(hr_image)
        lr_vis = z_scale_image(lr_image)

        # 可视化 HR patch（带 StarFinder 标记）
        ax_hr_marked = axes[i, 0] if num_selected > 1 else axes[0]
        ax_hr_marked.imshow(hr_vis, cmap='gray', origin='lower')
        ax_hr_marked.set_title(f'HR Patch {i} (Marked) @ {hr_coord}')
        for star in hr_stars:
            ax_hr_marked.scatter(star['rel_x'], star['rel_y'], c='lime', s=50, marker='x')  # 绿色 X 标记
        ax_hr_marked.axis('off')

        # 可视化 LR patch（带映射的 StarFinder 标记）
        ax_lr = axes[i, 1] if num_selected > 1 else axes[1]
        ax_lr.imshow(lr_vis, cmap='gray', origin='lower')
        ax_lr.set_title(f'LR Patch {i} @ {lr_coord}')
        for star in lr_stars:
            ax_lr.scatter(star['rel_x'], star['rel_y'], c='lime', s=50, marker='x')  # 绿色 X 标记
        ax_lr.axis('off')

        # 可视化 HR patch（不带 StarFinder 标记）
        ax_hr_unmarked = axes[i, 2] if num_selected > 1 else axes[2]
        ax_hr_unmarked.imshow(hr_vis, cmap='gray', origin='lower')
        ax_hr_unmarked.set_title(f'HR Patch {i} (Unmarked) @ {hr_coord}')
        ax_hr_unmarked.axis('off')

    plt.tight_layout()
    plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/data_process/meter/vis/patch_visualization_with_unmarked.png')
    plt.show()

if __name__ == "__main__":
    hr_patch_dir = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset/train_hr_patch"
    lr_patch_dir = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset/train_lr_patch"
    visualize_patch_pairs(hr_patch_dir, lr_patch_dir, num_pairs=5)  # 可视化前5对符合条件的 patch
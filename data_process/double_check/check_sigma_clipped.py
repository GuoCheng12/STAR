import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval

# 读取 FITS 文件并应用 3-sigma clipping
def load_and_clip_fits(file_path):
    """加载 FITS 文件，返回原始图像和 3-sigma clipped 图像"""
    with fits.open(file_path) as hdul:
        image = hdul[1].data.astype(float)  # SCI 数据从 HDU 1 读取
        
        # 3-sigma clipping
        mean = np.nanmean(image)  # 计算均值，忽略 NaN
        sigma = np.nanstd(image)  # 计算标准差，忽略 NaN
        lower_bound = mean - 10 * sigma
        upper_bound = mean + 10 * sigma
        image_clipped = image.copy()
        image_clipped[(image < lower_bound) | (image > upper_bound)] = np.nan
        
        return image, image_clipped

# 可视化原始和裁剪后的图像
def visualize_clipping(original, clipped, output_path):
    """可视化原始图像和 3-sigma clipped 图像"""
    # 使用 Z-scale 归一化
    z = ZScaleInterval()
    vmin_orig, vmax_orig = z.get_limits(original)
    vmin_clip, vmax_clip = z.get_limits(clipped)
    
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原始图像
    axes[0].imshow(original, cmap='gray', vmin=vmin_orig, vmax=vmax_orig)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 显示裁剪后的图像
    axes[1].imshow(clipped, cmap='gray', vmin=vmin_clip, vmax=vmax_clip)
    axes[1].set_title("3-Sigma Clipped Image")
    axes[1].axis('off')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 指定测试文件路径
    fits_file_path = "/home/bingxing2/ailab/group/ai4astro/Datasets/Astro_SR/origin_fits/hst_14182_17_acs_wfc_f814w_jcwr17_drc.fits.gz"
    output_image_path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/clipping_test.png"
    
    # 加载和裁剪图像
    original_image, clipped_image = load_and_clip_fits(fits_file_path)
    
    # 打印统计信息
    print(f"Original mean: {np.nanmean(original_image):.2f}, sigma: {np.nanstd(original_image):.2f}")
    print(f"Clipped mean: {np.nanmean(clipped_image):.2f}, sigma: {np.nanstd(clipped_image):.2f}")
    
    # 可视化效果
    visualize_clipping(original_image, clipped_image, output_image_path)
    print(f"可视化结果已保存至 {output_image_path}")
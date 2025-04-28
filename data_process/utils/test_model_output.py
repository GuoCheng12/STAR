import numpy as np
import matplotlib.pyplot as plt
import sep
from astropy.stats import sigma_clipped_stats

def load_data(npy_path):
    """加载 .npy 文件中的 pred 和 hr 图像"""
    data = np.load(npy_path, allow_pickle=True).item()
    pred = data['pred'].squeeze()
    hr = data['gt'].squeeze()
    return pred, hr

def visualize_images(pred, hr):
    """可视化 pred 和 hr 图像"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(pred, cmap='gray')
    axes[0].set_title('Predicted Image')
    axes[0].set_axis_off()
    axes[1].imshow(hr, cmap='gray')
    axes[1].set_title('Ground Truth Image')
    axes[1].set_axis_off()
    plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/origin.png')

def perform_photometry(image, mask):
    """对图像进行测光分析"""
    # 背景减除
    bkg = sep.Background(image, mask=~mask, bw=64, bh=64, fw=5, fh=5)
    image_sub = image - bkg.back()
    
    # 计算统计量
    valid_pixels = image_sub[mask]
    mean, median, std = sigma_clipped_stats(valid_pixels, sigma=3.0)
    
    # 设置检测阈值
    threshold = 1.5 * bkg.globalrms
    
    # 源检测
    sources = sep.extract(image_sub, 1.5, err=bkg.rms(), mask=~mask)
    print(f"检测到 {len(sources)} 个源。")
    
    # 测光
    flux, fluxerr, flag = sep.sum_ellipse(image_sub, sources['x'], sources['y'],
                                          sources['a'], sources['b'], sources['theta'],
                                          2.5, err=bkg.globalrms)
    valid_idx = ~np.isnan(flux)
    sources = sources[valid_idx]
    flux_cleaned = flux[valid_idx]
    
    # 生成注意力图
    attn_map = generate_attn_map(image_sub.shape, sources, flux_cleaned)
    # 应用掩码
    attn_map = np.where(mask, attn_map, np.nan)
    
    return sources, flux_cleaned, attn_map

def generate_attn_map(image_shape, sources, flux):
    attn_map = np.zeros(image_shape, dtype=np.float32)
    
    for i in range(len(sources)):
        x, y = sources['x'][i], sources['y'][i]
        a, b = sources['a'][i], sources['b'][i]
        theta = sources['theta'][i]
        
        # 生成椭圆掩码
        yy, xx = np.ogrid[:image_shape[0], :image_shape[1]]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 旋转后的坐标
        x_rot = (xx - x) * cos_theta + (yy - y) * sin_theta
        y_rot = -(xx - x) * sin_theta + (yy - y) * cos_theta
        
        # 椭圆方程
        ellipse_mask = (x_rot / a)**2 + (y_rot / b)**2 <= 1
        
        # 将 flux[i] 取绝对值后分配到椭圆区域
        abs_flux = np.abs(flux[i])  # 取绝对值
        attn_map[ellipse_mask] += abs_flux  # 叠加绝对流量
    
    return attn_map

def main(npy_path):
    """主函数"""
    # 加载数据
    pred, hr = load_data(npy_path)
    
    # 可视化图像
    visualize_images(pred, hr)
    
    # 示例掩码（根据实际需求替换）
    mask = np.ones_like(pred, dtype=bool)
    
    # 对 pred 进行测光
    print("Processing Predicted Image...")
    sources_pred, flux_pred, attn_map_pred = perform_photometry(pred, mask)
    
    # 对 hr 进行测光
    print("Processing Ground Truth Image...")
    sources_hr, flux_hr, attn_map_hr = perform_photometry(hr, mask)
    
    # 可选：可视化注意力图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(attn_map_pred, cmap='viridis')
    axes[0].set_title('Predicted Attention Map')
    axes[0].set_axis_off()
    axes[1].imshow(attn_map_hr, cmap='viridis')
    axes[1].set_title('Ground Truth Attention Map')
    axes[1].set_axis_off()
    plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/attn_map.png')
    plt.close()
if __name__ == "__main__":
    npy_path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/data_process/meter/model_out/pred_img_batch0_img0.npy"  # 替换为实际文件路径
    main(npy_path)
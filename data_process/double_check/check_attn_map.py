import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pdb
def load_npy_data(npy_path):
    """加载 .npy 文件中的数据"""
    data = np.load(npy_path, allow_pickle=True).item()
    image = data['image']  # 256x256的HR图像
    stars = data['stars']  # 恒星信息列表
    return image, stars

def generate_attn_image(image_shape, stars, sigma=1.274):
    """生成注意力图像"""
    attn_image = np.zeros(image_shape)
    for star in stars:

        x, y = star['rel_x'], star['rel_y']  # 恒星位置
        flux = star['flux']  # 恒星通量
        # 创建高斯核
        cov = [[sigma**2, 0], [0, sigma**2]]
        gauss = multivariate_normal(mean=[x, y], cov=cov)
        # 生成网格
        x_grid, y_grid = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        pos = np.dstack((x_grid, y_grid))
        # 计算高斯权重
        weights = gauss.pdf(pos)
        # 归一化并缩放
        weights /= weights.max()
        weights *= flux
        # 累加到注意力图像
        attn_image += weights
    return attn_image

def visualize(image, attn_image, output_path):
    """可视化原始图像和注意力图像"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original HR Image')
    axes[0].axis('off')
    axes[1].imshow(attn_image, cmap='hot')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    npy_path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/train_hr_patch/hst_9405_2a_acs_wfc_f814w_j8iy2a_drc_padded_hr_hr_patch_19.npy"
    output_path = "visualization.png"
    # 加载数据
    image, stars = load_npy_data(npy_path)
    
    # 生成注意力图像，sigma固定为1.274
    attn_image = generate_attn_image((256, 256), stars, sigma=1.274)
    
    # 可视化
    visualize(image, attn_image, output_path)
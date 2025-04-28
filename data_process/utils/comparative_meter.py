import sep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def load_data(npy_path):
    """加载 .npy 文件中的 pred 和 hr 图像"""
    data = np.load(npy_path, allow_pickle=True).item()
    pred = data['pred'].squeeze()
    hr = data['gt'].squeeze()
    return pred, hr

def measure_flux_with_gt_sources(image, sources, mask=None):
    """
    使用 gt 图像的源信息对图像进行测光，返回源的流通量。
    
    参数:
    image: 2D numpy 数组，图像数据。
    sources: 结构化数组，包含源的位置和形状信息（来自 gt 图像）。
    mask: 2D numpy 数组，掩码（True 表示有效区域），可选。
    
    返回:
    flux: numpy 数组，源的流通量。
    """
    if mask is None:
        bkg = sep.Background(image)
    else:
        bkg = sep.Background(image, mask=~mask)
    image_sub = image - bkg.back()
    
    flux, fluxerr, flag = sep.sum_ellipse(
        image_sub, sources['x'], sources['y'],
        sources['a'], sources['b'], sources['theta'],
        2.5, err=bkg.globalrms
    )
    
    valid_idx = ~np.isnan(flux)
    flux_cleaned = flux[valid_idx]
    
    return flux_cleaned

def visualize_sources(gt, pred_img, sources):
    """生成两张子图，分别显示 gt 和 pred_img 上的源标识"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 显示 gt 图像及源标识
    ax1.imshow(gt, cmap='gray', origin='lower')
    ax1.set_title("GT Image with Detected Sources")
    for i in range(len(sources)):
        e = Ellipse(xy=(sources['x'][i], sources['y'][i]),
                    width=6 * sources['a'][i],
                    height=6 * sources['b'][i],
                    angle=sources['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax1.add_artist(e)
    
    # 显示 pred_img 图像及 gt 源标识
    ax2.imshow(pred_img, cmap='gray', origin='lower')
    ax2.set_title("Pred Image with GT Sources")
    for i in range(len(sources)):
        e = Ellipse(xy=(sources['x'][i], sources['y'][i]),
                    width=6 * sources['a'][i],
                    height=6 * sources['b'][i],
                    angle=sources['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax2.add_artist(e)
    
    plt.tight_layout()
    plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/verify_attn_map.png')
    plt.close()

# 加载图像数据
npy_path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/data_process/meter/model_out/pred_img_batch0_img0.npy"
pred_img, gt = load_data(npy_path)
mask_gt = ~np.isnan(gt)
mask_pred_img = ~np.isnan(pred_img)

# 在 gt 图像上检测源
bkg_gt = sep.Background(gt)
gt_sub = gt - bkg_gt.back()
sources_gt = sep.extract(gt_sub, 1.5, err=bkg_gt.rms(), mask=~mask_gt)

# 在 gt 和 pred_img 上测光
flux_gt = measure_flux_with_gt_sources(gt, sources_gt, mask=mask_gt)
flux_pred = measure_flux_with_gt_sources(pred_img, sources_gt, mask=mask_pred_img)

# 计算流通量一致性损失
flux_diff = np.abs(flux_pred - flux_gt)
loss = np.mean(flux_diff)

# 输出结果
print(f"GT 图像检测到的源数量: {len(sources_gt)}")
print(f"Pred 图像使用 GT 源测量的流通量数量: {len(flux_pred)}")
print(f"流通量一致性损失: {loss}")

# 可视化源标识
visualize_sources(gt, pred_img, sources_gt)
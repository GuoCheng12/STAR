import numpy as np
import sep
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from scipy.stats import multivariate_normal
from astropy.visualization import ZScaleInterval
import pdb
# 加载数据
path = '/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy/train_hr_patch/hst_12209_15_acs_wfc_f814w_jbiv15_drc_padded_hr_hr_patch_668.npy'
data = np.load(path, allow_pickle=True).item()
image = data['image'].astype(np.float32)
mask = data['mask'] 

# # Visualization
# pdb.set_trace()
# zscale = ZScaleInterval()
# vmin, vmax = zscale.get_limits(image)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), dpi=300)

# # Subplot 1: Original Image
# ax1.imshow(image, interpolation='nearest', cmap='gray',
#            vmin=vmin, vmax=vmax, origin='lower')
# ax1.set_title('Original Image')
# plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/origin.png')
image_cleaned = np.where(mask, image, 0.0)
def generate_attn_map(image_shape, sources, flux):
    """生成注意力图"""
    attn_map = np.zeros(image_shape, dtype=np.float32)
    for i in range(len(sources)):
        x, y = sources['x'][i], sources['y'][i]
        a, b = sources['a'][i], sources['b'][i]
        theta = sources['theta'][i]
        cov = np.array([[a**2, 0], [0, b**2]])
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rot_matrix = np.array([[cos_theta, -sin_theta], 
                                [sin_theta, cos_theta]])
        cov_rot = rot_matrix @ cov @ rot_matrix.T
        gauss = multivariate_normal(mean=[x, y], cov=cov_rot)
        x_grid, y_grid = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        pos = np.dstack((x_grid, y_grid))
        weights = gauss.pdf(pos)
        abs_flux = np.abs(flux[i])
        attn_map += weights * abs_flux
    return attn_map
bkg = sep.Background(image_cleaned, mask=~mask, bw=64, bh=64, fw=3, fh=3)
image_sub = image_cleaned - bkg.back()
valid_pixels = image_sub[mask]
mean, median, std = sigma_clipped_stats(valid_pixels, sigma=3.0)

threshold = 1.5 * bkg.globalrms

sources = sep.extract(image_sub, 1.5, err=bkg.rms(), mask=~mask)
print(f"检测到 {len(sources)} 个源。")
# 步骤 4：测光
flux, fluxerr, flag = sep.sum_ellipse(image_sub, sources['x'], sources['y'],
                                      sources['a'], sources['b'], sources['theta'],
                                      2.5, err=bkg.globalrms)
valid_idx = ~np.isnan(flux)
sources = sources[valid_idx]
flux_cleaned = flux[valid_idx]
attn_map = generate_attn_map(image_sub.shape, sources, flux_cleaned)
# 应用 mask
attn_map = np.where(mask, attn_map, np.nan)

# 保存 attn_map 到原文件
data['attn_map'] = attn_map
np.save('/home/bingxing2/ailab/scxlab0061/Astro_SR/data_process/dataset_tmp/hst_16278_46_acs_wfc_f814w_jecd46_drc_padded_hr_hr_patch_422.npy', data)

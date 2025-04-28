import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import pdb
# 加载 .npy 文件
patch_data = np.load('/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy/train_hr_patch/hst_9405_2a_acs_wfc_f814w_j8iy2a_drc_padded_hr_hr_patch_0.npy', allow_pickle=True).item()
# 提取图像和掩码
image = patch_data['image']
mask = patch_data['mask']
norm = ImageNormalize(image, interval=ZScaleInterval())
# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 显示图像
axes[0].imshow(image, cmap='gray',origin='lower', norm=norm)
axes[0].set_title('Image Patch')

# 显示掩码
axes[1].imshow(mask, cmap='gray', origin='lower')
axes[1].set_title('Mask Patch')

# 显示图像
plt.savefig("/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/vis_npy_lr.png")
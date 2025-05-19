import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import json
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clip

# FITS 文件路径
fits_filepath = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_real_world/hst_10152_05_acs_wfc_f814w_j90i05_drc_padded_hr.fits"

# 打开 FITS 文件并获取图像数据
hdu = fits.open(fits_filepath)[0]
print(fits.info(fits_filepath))  # 显示 FITS 文件信息
img_data = hdu.data

# bkg_estimator = MedianBackground()  # 使用中位数作为背景估计器
# bkg = Background2D(img_data, (64, 64), filter_size=(3, 3), bkg_estimator=bkg_estimator)
# img_data_bkg_subtracted = img_data - bkg.background  # 减去背景

# 应用 sigma clipping
#clipped_data = sigma_clip(img_data_bkg_subtracted, sigma=10.0, cenfunc='median', stdfunc='std', maxiters=5)

# 将被剔除的像素设置为 NaN
#img_data_clipped = np.where(clipped_data.mask, np.nan, img_data_bkg_subtracted)

# 使用 ZScaleInterval 进行归一化
norm = ImageNormalize(img_data, interval=ZScaleInterval())

# 可视化
plt.figure(figsize=(16, 16))
plt.imshow(img_data, cmap='gray',  origin='lower', norm=norm)
plt.colorbar(label='Intensity')
plt.title(f"FITS Image with Sigma Clipping: {fits_filepath}")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/test1.png')

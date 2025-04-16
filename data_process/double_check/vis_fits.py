import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import json

# 文件路径
fits_filepath = "/home/bingxing2/ailab/scxlab0061/Astro_SR/data_process/meter/dataset/psf_hr/hst_15851_13_acs_wfc_f814w_je5613_drc_padded_hr.fits"
json_path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/data_process/meter/dataset/psf_hr/hst_15851_13_acs_wfc_f814w_je5613_drc_stars.json"

# 加载 FITS 文件
hdu = fits.open(fits_filepath)[0]
print(fits.info(fits_filepath))  # 打印 FITS 文件信息
img_data = hdu.data.astype(float)  # 读取图像数据并转换为浮点型

# 加载恒星信息
with open(json_path, 'r') as f:
    stars = json.load(f)

# Z-scale 归一化
norm = ImageNormalize(img_data, interval=ZScaleInterval())

# 可视化
plt.figure(figsize=(10, 10))
plt.imshow(img_data, cmap='gray', norm=norm, origin='lower')
plt.title('FITS Image with Detected Stars')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')

# 标记恒星位置
for star in stars:
    x = star['x']
    y = star['y']
    plt.scatter(x, y, c='lime', s=5, label='Stars' if star == stars[0] else None)

plt.legend()
plt.colorbar(label='Intensity')
plt.tight_layout()
plt.savefig("/home/bingxing2/ailab/scxlab0061/Astro_SR/data_process/meter/vis/fits_image_with_stars.png")
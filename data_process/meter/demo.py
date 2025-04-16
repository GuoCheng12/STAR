import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from photutils.psf import  PSFPhotometry, CircularGaussianPRF, SourceGrouper, IntegratedGaussianPRF
from photutils.detection import DAOStarFinder
from matplotlib.patches import Circle
from photutils.background import Background2D, MedianBackground
import pdb
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.wcs import WCS
from reproject import reproject_exact


def downsample_image(image, wcs, scale_factor=2):
    """下采样图像并更新 WCS"""
    target_shape = (int(image.shape[0] / scale_factor), int(image.shape[1] / scale_factor))
    target_wcs = wcs.deepcopy()
    target_wcs.wcs.crpix = [crpix / scale_factor for crpix in wcs.wcs.crpix]
    if hasattr(wcs.wcs, 'cd'):
        target_wcs.wcs.cd = wcs.wcs.cd * scale_factor
    else:
        target_wcs.wcs.cdelt = [cdelt * scale_factor for cdelt in wcs.wcs.cdelt]
    downsampled_image, _ = reproject_exact((image, wcs), target_wcs, shape_out=target_shape)
    downsampled_image = scale_factor**2 * downsampled_image  # 修正通量
    return downsampled_image, target_wcs


# 1. Load the FITS image
filename = "/home/bingxing2/ailab/group/ai4astro/Datasets/Astro_SR/origin_fits/hst_15851_13_acs_wfc_f814w_je5613_drc.fits"
hdul = fits.open(filename)
data_hr = hdul[1].data.astype(float)  # 读取图像数据
wcs_hr = WCS(hdul[1].header)  # 读取WCS信息
hdul.close()


patch_size = 256  # 假设patch_size为256
h, w = data_hr.shape
pad_h = (patch_size - h % patch_size) % patch_size
pad_w = (patch_size - w % patch_size) % patch_size
padded_hr = np.pad(data_hr, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=np.nan)

padded_wcs = wcs_hr.deepcopy()
padded_wcs.array_shape = padded_hr.shape  # 更新WCS的图像尺寸
mask = ~np.isnan(padded_hr)

# 2. Subtract the background
bkg = Background2D(padded_hr, (64, 64), filter_size=(3, 3), bkg_estimator=MedianBackground(), mask=~mask)
data_hr_sub = padded_hr - bkg.background
data_hr_sub_masked = np.where(mask, data_hr_sub, np.nan)  # 掩盖无效区域

# 3. Detect stars
mean, median, std = sigma_clipped_stats(data_hr_sub_masked, sigma=3.0)
daofind = DAOStarFinder(fwhm=3.0, threshold=8.0 * std)
sources_tbl = daofind(data_hr_sub_masked)

if sources_tbl is None or len(sources_tbl) == 0:
    print("No stars detected! Adjust detection parameters.")
    exit()

# 重命名 sources_tbl 中的列
sources_tbl.rename_column('xcentroid', 'x_0')
sources_tbl.rename_column('ycentroid', 'y_0')
sources_tbl.rename_column('flux', 'flux_0')

# 4. Perform PSF photometry
psf_model = IntegratedGaussianPRF(sigma=1.0)  # Initial sigma value
psf_model.sigma.fixed = False   # Allow sigma to be fitted
photometry = PSFPhotometry(psf_model, fit_shape=(13, 13), aperture_radius=3.0)
psf_phot_results = photometry(data_hr_sub_masked, init_params=sources_tbl[['x_0', 'y_0', 'flux_0']])

# 5. Downsample the image
scale_factor = 2
downsampled_image, target_wcs = downsample_image(padded_hr, padded_wcs, scale_factor=scale_factor)

# 6. Convert PSF results to DataFrame for easier iteration
psf_phot_results_df = psf_phot_results.to_pandas()

# 7. Visualize HR and LR images with star positions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# HR image visualization
ax1.imshow(data_hr_sub, origin='lower', cmap='gray')
ax1.set_title('HR Image with Detected Stars')
for idx, row in psf_phot_results_df.iterrows():
    x_hr = row['x_fit']
    y_hr = row['y_fit']
    ax1.plot(x_hr, y_hr, 'ro', markersize=5)  # Red dots for star positions

# LR image visualization
ax2.imshow(downsampled_image, origin='lower', cmap='gray')
ax2.set_title('LR Image with Mapped Star Positions')
for idx, row in psf_phot_results_df.iterrows():
    x_hr = row['x_fit']
    y_hr = row['y_fit']
    x_lr = x_hr / scale_factor  # Map HR coordinates to LR
    y_lr = y_hr / scale_factor
    ax2.plot(x_lr, y_lr, 'ro', markersize=5)  # Red dots for mapped star positions

plt.tight_layout()
plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/result_with_lr.png')

# 8. Output star positions (based on padded HR coordinates)
print("Detected stars (x_fit, y_fit):")
for idx, row in psf_phot_results_df.iterrows():
    print(f"Star {idx}: x={row['x_fit']:.2f}, y={row['y_fit']:.2f}")
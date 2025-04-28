import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry, CircularGaussianPRF
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel, convolve
from reproject import reproject_exact
import matplotlib.patches as patches
from tqdm import tqdm
import json
from astropy.visualization import ZScaleInterval
import pdb
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.modeling.functional_models import Gaussian2D

def generate_psf(sigma=1.2, size=15):
    """生成高斯 PSF 核"""
    half_size = size // 2
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    x, y = np.meshgrid(x, y)
    psf = Gaussian2D(amplitude=1.0, x_mean=0, y_mean=0, x_stddev=sigma, y_stddev=sigma)(x, y)
    return psf / psf.sum()

def downsample_image(image, wcs, scale_factor=2):
    """使用 reproject_exact 下采样图像并更新 WCS"""
    target_shape = (int(image.shape[0] / scale_factor), int(image.shape[1] / scale_factor))
    target_wcs = wcs.deepcopy()
    target_wcs.wcs.crpix = [crpix / scale_factor for crpix in wcs.wcs.crpix]
    if hasattr(wcs.wcs, 'cd'):
        target_wcs.wcs.cd = [cd * scale_factor for cd in wcs.wcs.cd]
    else:
        target_wcs.wcs.cdelt = [cdelt * scale_factor for cdelt in wcs.wcs.cdelt]
    target_wcs.array_shape = target_shape
    downsampled_image, _ = reproject_exact((image, wcs), target_wcs, shape_out=target_shape)
    downsampled_image = scale_factor**2 * downsampled_image
    return downsampled_image, target_wcs

def apply_psf(image, psf, mask=None):
    """应用 PSF 模糊，处理 NaN 值并归一化 PSF"""
    if mask is None:
        mask = ~np.isnan(image)
    image_temp = np.where(mask, image, 0.0)
    blurred_image = convolve(image_temp, psf, normalize_kernel=True)
    blurred_image[~mask] = np.nan
    return blurred_image

def generate_lr_image(hr_image, hr_wcs, psf_sigma=1.5, scale_factor=2):
    """生成低分辨率图像（PSF模糊 + 下采样）"""
    psf = generate_psf(sigma=psf_sigma)
    blurred = apply_psf(hr_image, psf)
    lr_image, lr_wcs = downsample_image(blurred, hr_wcs, scale_factor=scale_factor)
    return lr_image, lr_wcs

def apply_sigma_clipping(image, sigma=10):
    """应用sigma clipping去除图像中的异常值"""
    mean = np.nanmean(image)
    std = np.nanstd(image)
    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std
    image_clipped = image.copy()
    image_clipped[(image < lower_bound) | (image > upper_bound)] = np.nan
    return image_clipped

def perform_psf_photometry(image, mask):
    """在图像上进行PSF测光，返回恒星的坐标和通量信息"""
    bkg = Background2D(image, (64, 64), filter_size=(3, 3), 
                       bkg_estimator=MedianBackground(), mask=~mask)
    data_sub = image - bkg.background
    data_sub_masked = np.where(mask, data_sub, np.nan)

    mean, median, std = sigma_clipped_stats(data_sub_masked, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=8.0 * std)
    sources_tbl = daofind(data_sub_masked)

    if sources_tbl is None or len(sources_tbl) == 0:
        print("未在图像中检测到恒星！")
        return []

    sources_tbl.rename_column('xcentroid', 'x_0')
    sources_tbl.rename_column('ycentroid', 'y_0')
    sources_tbl.rename_column('flux', 'flux_0')
    psf_model = CircularGaussianPRF(fwhm=3.0)
    psf_model.x_0.fixed = False
    psf_model.y_0.fixed = False
    psf_model.flux.fixed = False
    psf_model.fwhm.fixed = True
    photometry = PSFPhotometry(psf_model, fit_shape=(13, 13), aperture_radius=7.0)
    phot_table = photometry(data_sub_masked, init_params=sources_tbl[['x_0', 'y_0', 'flux_0']])

    stars = []
    for row in phot_table:
        x = float(row['x_fit'])
        y = float(row['y_fit'])
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[int(np.round(y)), int(np.round(x))]:
            stars.append({'x': x, 'y': y, 'flux': float(row['flux_fit'])})
    return stars

def perform_psf_photometry_lr(lr_image, lr_mask, hr_stars, lr_wcs, padded_wcs, scale_factor=2):
    """在LR图像上进行PSF测光，返回LR测光结果和匹配的HR子集"""
    bkg = Background2D(lr_image, (64 // scale_factor, 64 // scale_factor), filter_size=(3, 3), 
                       bkg_estimator=MedianBackground(), mask=~lr_mask)
    data_sub = lr_image - bkg.background
    data_sub_masked = np.where(lr_mask, data_sub, np.nan)

    positions = []
    for star in hr_stars:
        ra, dec = padded_wcs.pixel_to_world_values(star['x'], star['y'])
        x_lr, y_lr = lr_wcs.world_to_pixel_values(ra, dec)
        positions.append((x_lr, y_lr))

    flux_init = [star['flux'] for star in hr_stars]
    lr_stars_init = [{'x_init': pos[0], 'y_init': pos[1], 'flux_init': flux} 
                     for pos, flux in zip(positions, flux_init)]
    init_params = Table(rows=lr_stars_init, names=('x_init', 'y_init', 'flux_init'))

    psf_model = CircularGaussianPRF(fwhm=3.0 / scale_factor)
    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    psf_model.flux.fixed = False
    psf_model.fwhm.fixed = True

    fit_shape = (13 // scale_factor if 13 // scale_factor % 2 == 1 else 13 // scale_factor + 1, 
                 13 // scale_factor if 13 // scale_factor % 2 == 1 else 13 // scale_factor + 1)
    aperture_radius_psf = 7.0 / scale_factor

    photometry = PSFPhotometry(psf_model, fit_shape=fit_shape, aperture_radius=aperture_radius_psf)
    phot_table = photometry(data_sub_masked, mask=~lr_mask, init_params=init_params)

    lr_stars = []
    matched_hr_stars = []
    for i, row in enumerate(phot_table):
        x = float(row['x_fit'])
        y = float(row['y_fit'])
        if 0 <= y < lr_mask.shape[0] and 0 <= x < lr_mask.shape[1] and lr_mask[int(y), int(x)]:
            lr_stars.append({'x': x, 'y': y, 'flux': float(row['flux_fit'])})
            matched_hr_stars.append(hr_stars[i])

    filtered_lr_stars = len(init_params) - len(lr_stars)
    print(f"Number of LR stars filtered out: {filtered_lr_stars}")
    return lr_stars, matched_hr_stars

def z_scale_image(image):
    """对图像应用 Z-scale 归一化"""
    z = ZScaleInterval()
    vmin, vmax = z.get_limits(image)
    normalized_image = np.clip((image - vmin) / (vmax - vmin), 0, 1)
    return normalized_image

def visualize_photometry(hr_image, lr_image, hr_stars, lr_stars):
    """在HR和LR大图上可视化测光结果，颜色根据通量差异"""
    flux_diffs = [lr_star['flux'] - hr_star['flux'] for hr_star, lr_star in zip(hr_stars, lr_stars)]

    hr_image_z = z_scale_image(hr_image)
    lr_image_z = z_scale_image(lr_image)

    # HR图像可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(hr_image_z, cmap='gray', origin='lower')
    for hr_star, flux_diff in zip(hr_stars, flux_diffs):
        color = 'green' if flux_diff > 0 else 'red'
        plt.scatter(hr_star['x'], hr_star['y'], s=8, c=color, marker='o', edgecolor='none')
    plt.title('HR Image with Photometry Results')
    plt.savefig('hr_photometry.png', dpi=150, bbox_inches='tight')
    plt.close()

    # LR图像可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(lr_image_z, cmap='gray', origin='lower')
    for lr_star, flux_diff in zip(lr_stars, flux_diffs):
        color = 'green' if flux_diff > 0 else 'red'
        plt.scatter(lr_star['x'], lr_star['y'], s=8, c=color, marker='o', edgecolor='none')
    plt.title('LR Image with Photometry Results')
    plt.savefig('lr_photometry.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """主函数，执行FITS图像测光和可视化"""
    fits_file = "/home/bingxing2/ailab/group/ai4astro/Datasets/zgh/origin_fits/hst_10152_05_acs_wfc_f814w_j90i05_drc.fits.gz"
    with fits.open(fits_file) as hdul:
        hr_image = hdul[1].data.astype(float)
        hr_wcs = WCS(hdul[1].header)

    hr_image_clipped = apply_sigma_clipping(hr_image, sigma=10)
    print(f"HR image mean: {np.nanmean(hr_image_clipped):.6f}, std: {np.nanstd(hr_image_clipped):.6f}, min: {np.nanmin(hr_image_clipped):.6f}, max: {np.nanmax(hr_image_clipped):.6f}")
    mask = ~np.isnan(hr_image_clipped)

    pad_y = (256 - hr_image_clipped.shape[0] % 256) % 256
    pad_x = (256 - hr_image_clipped.shape[1] % 256) % 256
    hr_padded = np.pad(hr_image_clipped, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=np.nan)
    mask_padded = np.pad(mask, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=False)

    padded_wcs = hr_wcs.deepcopy()
    padded_wcs.array_shape = hr_padded.shape

    hr_stars = perform_psf_photometry(hr_padded, mask_padded)
    lr_image, lr_wcs = generate_lr_image(hr_padded, padded_wcs, psf_sigma=1.2)
    pdb.set_trace()
    lr_mask = ~np.isnan(lr_image)

    lr_stars, matched_hr_stars = perform_psf_photometry_lr(lr_image, lr_mask, hr_stars, lr_wcs, padded_wcs)

    print(f"Number of HR stars (matched): {len(matched_hr_stars)}")
    print(f"Number of LR stars: {len(lr_stars)}")

    visualize_photometry(hr_padded, lr_image, matched_hr_stars, lr_stars)

if __name__ == "__main__":
    main()
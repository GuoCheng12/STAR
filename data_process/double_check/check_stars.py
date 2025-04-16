import numpy as np
from astropy.io import fits
from photutils.detection import DAOStarFinder
from photutils.psf import IntegratedGaussianPRF, PSFPhotometry
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt

# 加载 FITS 文件
def load_fits(file_path):
    with fits.open(file_path) as hdul:
        image = hdul[1].data.astype(float)  # 假设图像数据在 HDU 1
    return image

# PSF 测光函数
def perform_psf_photometry(image, mask):
    bkg = Background2D(image, (64, 64), filter_size=(3, 3), bkg_estimator=MedianBackground(), mask=~mask)
    data_sub = image - bkg.background
    data_sub_masked = np.where(mask, data_sub, np.nan)

    mean, median, std = sigma_clipped_stats(data_sub_masked, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=8.0 * std)
    sources_tbl = daofind(data_sub_masked)

    if sources_tbl is None or len(sources_tbl) == 0:
        print("No stars detected in this image!")
        return []

    sources_tbl.rename_column('xcentroid', 'x_0')
    sources_tbl.rename_column('ycentroid', 'y_0')
    sources_tbl.rename_column('flux', 'flux_0')

    psf_model = IntegratedGaussianPRF(sigma=1.0)
    psf_model.sigma.fixed = False
    photometry = PSFPhotometry(psf_model, fit_shape=(11, 11), aperture_radius=5.0)
    phot_table = photometry(data_sub_masked, init_params=sources_tbl[['x_0', 'y_0', 'flux_0']])

    stars = []
    for row in phot_table:
        star = {
            'x': float(row['x_fit']),
            'y': float(row['y_fit']),
            'flux': float(row['flux_fit'])
        }
        stars.append(star)
    return stars

# 图像填充函数
def pad_to_multiple(image, multiple=256, pad_value=np.nan):
    h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_value)
    return padded_image

# 主函数：验证 stars 和 new_stars 的位置
def verify_star_positions(origin_image_path):
    # 加载原始图像
    origin_image = load_fits(origin_image_path)
    mask = ~np.isnan(origin_image)

    # 填充图像
    padded_image = pad_to_multiple(origin_image, multiple=256, pad_value=np.nan)
    padded_mask = ~np.isnan(padded_image)

    # 对填充后的图像进行 PSF 测光 (stars)
    stars = perform_psf_photometry(padded_image, padded_mask)
    # 对原始图像进行 PSF 测光 (new_stars)
    new_stars = perform_psf_photometry(origin_image, mask)

    # 输出恒星数量
    print(f"Number of stars in padded image: {len(stars)}")
    print(f"Number of stars in original image: {len(new_stars)}")

    # 计算填充的偏移量（假设填充在底部和右侧）
    pad_h = padded_image.shape[0] - origin_image.shape[0]
    pad_w = padded_image.shape[1] - origin_image.shape[1]
    pad_left = 0  # 左侧无填充
    pad_top = 0   # 顶部无填充

    # 调整 stars 的位置到原始图像坐标系
    adjusted_stars = []
    for star in stars:
        adjusted_x = star['x'] - pad_left
        adjusted_y = star['y'] - pad_top
        if 0 <= adjusted_x < origin_image.shape[1] and 0 <= adjusted_y < origin_image.shape[0]:
            adjusted_stars.append({'x': adjusted_x, 'y': adjusted_y, 'flux': star['flux']})

    print(f"Number of stars after adjustment: {len(adjusted_stars)}")

    # 比较 adjusted_stars 和 new_stars 的位置
    for i, (star_adj, star_new) in enumerate(zip(adjusted_stars, new_stars)):
        print(f"Star {i}:")
        print(f"  Padded image (adjusted): x={star_adj['x']:.2f}, y={star_adj['y']:.2f}, flux={star_adj['flux']:.2f}")
        print(f"  Original image: x={star_new['x']:.2f}, y={star_new['y']:.2f}, flux={star_new['flux']:.2f}")
        print(f"  Position difference: dx={star_adj['x'] - star_new['x']:.2f}, dy={star_adj['y'] - star_new['y']:.2f}")



if __name__ == "__main__":
    origin_image_path = "/home/bingxing2/ailab/group/ai4astro/Datasets/Astro_SR/origin_fits/hst_15851_13_acs_wfc_f814w_je5613_drc.fits"  # 请替换为您的图像路径
    verify_star_positions(origin_image_path)
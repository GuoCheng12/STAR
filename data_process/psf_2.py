import os
import random
import argparse
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve
from astropy.modeling.functional_models import Gaussian2D, AiryDisk2D
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.psf import CircularGaussianPRF, PSFPhotometry
from photutils.background import Background2D, MedianBackground
from reproject import reproject_exact
import numpy as np
from shapely.wkt import loads
import json
import pdb

# 生成 PSF 的函数
def generate_psf(psf_type=None, sigma=None, sigma_min=0.8, sigma_max=1.2, radius=None, radius_min=1.5, radius_max=1.9, size=15):
    """
    根据指定类型和参数生成 PSF，支持固定值或范围。
    参数：
        psf_type: 'gaussian' 或 'airy'，若为 None 则随机选择
        sigma: Gaussian PSF 的标准差（固定值）
        sigma_min, sigma_max: Gaussian PSF 的标准差范围
        radius: Airy PSF 的半径（固定值）
        radius_min, radius_max: Airy PSF 的半径范围
        size: PSF 核的大小
    返回：
        归一化的 PSF 核
    """
    if psf_type is None:
        psf_type = random.choice(['gaussian', 'airy'])
    
    half_size = size // 2
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    x, y = np.meshgrid(x, y)
    
    if psf_type == 'gaussian':
        if sigma is not None:
            print(f"Using fixed sigma = {sigma} for Gaussian PSF")
        else:
            sigma = random.uniform(sigma_min, sigma_max)
            print(f"Generated Gaussian PSF with sigma = {sigma} from range [{sigma_min}, {sigma_max}]")
        psf = Gaussian2D(amplitude=1.0, x_mean=0, y_mean=0, x_stddev=sigma, y_stddev=sigma)(x, y)
    elif psf_type == 'airy':
        if radius is not None:
            print(f"Using fixed radius = {radius} for Airy PSF")
        else:
            radius = random.uniform(radius_min, radius_max)
            print(f"Generated Airy PSF with radius = {radius} from range [{radius_min}, {radius_max}]")
        psf = AiryDisk2D(amplitude=1.0, x_0=0, y_0=0, radius=radius)(x, y)
    else:
        raise ValueError(f"Invalid PSF type: {psf_type}")
    
    return psf / psf.sum()

# 加载 FITS 文件并应用 10-sigma clipping
def load_fits(file_path):
    """加载 FITS 文件，应用 3-sigma clipping 去除噪声，返回图像数据、掩码和 WCS 信息"""
    with fits.open(file_path) as hdul:
        image = hdul[1].data.astype(float)  # SCI 数据从 HDU 1 读取
        wcs = WCS(hdul[1].header)
        mean = np.nanmean(image)  # 计算均值，忽略 NaN
        sigma = np.nanstd(image)  # 计算标准差，忽略 NaN
        lower_bound = mean - 10 * sigma
        upper_bound = mean + 10 * sigma
        image_clipped = image.copy()
        image_clipped[(image < lower_bound) | (image > upper_bound)] = np.nan

        mask = ~np.isnan(image_clipped)
        if np.all(mask):
            mask = np.ones_like(image_clipped, dtype=bool)

        if image_clipped.shape[0] > 6000 or image_clipped.shape[1] > 6000:
            raise ValueError(f"图像 shape {image_clipped.shape} 过大，跳过处理")
        
        return image_clipped, mask, wcs

# PSF 测光并记录恒星信息
def perform_psf_photometry(image, mask):
    """在图像上进行PSF测光，返回恒星的坐标和通量信息。"""
    # 计算背景并减去
    bkg = Background2D(image, (64, 64), filter_size=(3, 3), 
                      bkg_estimator=MedianBackground(), mask=~mask)
    data_sub = image - bkg.background
    data_sub_masked = np.where(mask, data_sub, np.nan)

    # 计算统计信息并检测恒星
    mean, median, std = sigma_clipped_stats(data_sub_masked, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=8.0 * std)
    sources_tbl = daofind(data_sub_masked)

    # 检查是否检测到恒星
    if sources_tbl is None or len(sources_tbl) == 0:
        print("未在图像中检测到恒星！")
        return []

    # 重命名列名以适配PSF测光
    sources_tbl.rename_column('xcentroid', 'x_0')
    sources_tbl.rename_column('ycentroid', 'y_0')
    sources_tbl.rename_column('flux', 'flux_0')

    # 执行PSF测光
    psf_model = CircularGaussianPRF(fwhm=3.0)
    psf_model.x_0.fixed = False
    psf_model.y_0.fixed = False
    psf_model.flux.fixed = False
    psf_model.fwhm.fixed = True

    photometry = PSFPhotometry(psf_model, fit_shape=(13, 13), aperture_radius=7.0)
    phot_table = photometry(data_sub_masked, init_params=sources_tbl[['x_0', 'y_0', 'flux_0']])

    # 提取恒星信息，并过滤mask值为False的恒星
    stars = []
    for row in phot_table:
        x = float(row['x_fit'])
        y = float(row['y_fit'])
        # 确保坐标在图像范围内
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            # 检查mask值，只保留mask值为True的恒星
            if mask[int(np.round(y)), int(np.round(x))]:
                star = {
                    'x': x,
                    'y': y,
                    'flux': float(row['flux_fit'])
                }
                stars.append(star)
    
    # 验证：检查恒星坐标是否在图像范围内
    invalid_stars = [star for star in stars if not (0 <= star['x'] < image.shape[1] and 0 <= star['y'] < image.shape[0])]
    if invalid_stars:
        print(f"警告：检测到 {len(invalid_stars)} 个恒星坐标超出图像范围！")

    return stars

# 应用 PSF 模糊
def apply_psf(image, psf, mask=None):
    """应用 PSF 模糊，mask 为 None 时使用图像默认掩码"""
    if mask is None:
        mask = ~np.isnan(image)
    image_temp = np.where(mask, image, 0.0)
    blurred_image = convolve(image_temp, psf, normalize_kernel=True)
    blurred_image[~mask] = np.nan
    return blurred_image

# Padding 图像到指定倍数
def pad_to_multiple(image, mask, wcs, multiple=256, pad_value=np.nan):
    """将图像和掩码 padding 到指定倍数，并更新 WCS"""
    h, w = image.shape

    pad_h = (multiple - h % multiple) % multiple  
    pad_w = (multiple - w % multiple) % multiple  
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_value)
    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)
    padded_wcs = wcs.deepcopy()
    padded_wcs.array_shape = padded_image.shape
    return padded_image, padded_mask, padded_wcs

# 下采样图像
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

# 保存下采样图像
def save_downsampled_image(image, wcs, output_dir, identifier):
    """保存下采样后的图像为 FITS 文件"""
    lr_path = os.path.join(output_dir, f"{identifier}_downsampled.fits")
    hdu = fits.PrimaryHDU(image, header=wcs.to_header())
    hdu.writeto(lr_path, overwrite=True)
    return lr_path

# 保存 padding 后的 HR 图像
def save_padded_hr_image(image, wcs, output_dir, identifier):
    """保存 padding 后的 HR 图像为 FITS 文件"""
    padded_hr_path = os.path.join(output_dir, f"{identifier}_padded_hr.fits")
    hdu = fits.PrimaryHDU(image, header=wcs.to_header())
    hdu.writeto(padded_hr_path, overwrite=True)
    return padded_hr_path

# 单个文件处理函数（单线程版本）
def process_single_file(file_data, lr_output_dir, hr_output_dir, scale_factor, psf_type, sigma, sigma_min, sigma_max, radius, radius_min, radius_max):
    fits_filepath, padded_image, padded_mask, padded_wcs = file_data
    try:
        identifier = os.path.basename(fits_filepath).replace(".fits", "").replace(".gz", "")
        # 在 padding 后的 HR 图像上进行 PSF 测光
        stars = perform_psf_photometry(padded_image, padded_mask)
        # new_stars = perform_psf_photometry(origin_image, mask)
        # print("Are the two star lists equal?", stars == new_stars)

        # 保存 PSF 测光结果（JSON 文件）
        stars_file = os.path.join(hr_output_dir, f"{identifier}_stars.json")
        with open(stars_file, 'w') as f:
            json.dump(stars, f, indent=4)
        
        # 保存 padding 后的 HR 图像
        padded_hr_path = save_padded_hr_image(padded_image, padded_wcs, hr_output_dir, identifier)
        
        # 生成 PSF 并应用模糊
        psf = generate_psf(psf_type, sigma, sigma_min, sigma_max, radius, radius_min, radius_max)
        blurred_image = apply_psf(padded_image, psf, mask=padded_mask)
        
        # 下采样生成 LR 图像
        downsampled_image, target_wcs = downsample_image(blurred_image, padded_wcs, scale_factor)
        
        # 保存 LR 图像
        lr_path = save_downsampled_image(downsampled_image, target_wcs, lr_output_dir, identifier)
        
        return f"{padded_hr_path},{lr_path},{stars_file}"
    except Exception as e:
        print(f"Processing {fits_filepath} fail: {e}")
        return None

# 主处理函数（单线程版本）
def process_fits_files(datasetlist_path, lr_output_dir, hr_output_dir, split_file_dir, scale_factor, psf_type, sigma, sigma_min, sigma_max, radius, radius_min, radius_max):
    os.makedirs(split_file_dir, exist_ok=True)
    train_files_path = os.path.join(split_file_dir, "train_files.txt")
    eval_files_path = os.path.join(split_file_dir, "eval_files.txt")
    os.makedirs(lr_output_dir, exist_ok=True)
    os.makedirs(hr_output_dir, exist_ok=True)
    
    if os.path.exists(train_files_path) and os.path.exists(eval_files_path):
        print("Detect existing train_files.txt and eval_files.txt and read them directly...")
        with open(train_files_path, "r") as f:
            train_files = [line.strip().split(',') for line in f.readlines()]
        with open(eval_files_path, "r") as f:
            eval_files = [line.strip().split(',') for line in f.readlines()]
    else:
        # 读取数据集并划分
        with open(datasetlist_path, "r") as f:
            lines = f.readlines()
        train_files = []
        eval_files = []
        for line in tqdm(lines, desc="Filtering and partitioning datasets"):
            try:
                fits_filepath, wkt_str = line.strip().split(":")
                with fits.open(fits_filepath) as hdul:
                    header = hdul[1].header
                    ncombine = header.get('NCOMBINE', 0)
                    if ncombine == 4:
                        polygon = loads(wkt_str)
                        ra_values = [point[0] for point in polygon.exterior.coords]
                        min_ra = min(ra_values)
                        max_ra = max(ra_values)
                        image, mask, wcs = load_fits(fits_filepath)
                        padded_image, padded_mask, padded_wcs = pad_to_multiple(image, mask, wcs, multiple=256, pad_value=np.nan)
                        if max_ra < 250:
                            train_files.append((fits_filepath, padded_image, padded_mask, padded_wcs))
                        elif min_ra > 255:
                            eval_files.append((fits_filepath, padded_image, padded_mask, padded_wcs))
            except ValueError as ve:
                print(f"skip {fits_filepath}: {ve}")
            except Exception as e:
                print(f"skip {fits_filepath} fail: {e}")

        # 处理训练集
        train_results = []
        for file_data in tqdm(train_files, desc="Processing train files"):
            result = process_single_file(file_data, lr_output_dir, hr_output_dir, scale_factor, psf_type, sigma, sigma_min, sigma_max, radius, radius_min, radius_max)
            if result:
                train_results.append(result)
        with open(train_files_path, "w") as f_train:
            for result in train_results:
                f_train.write(result + "\n")
        
        # 处理验证集
        eval_results = []
        for file_data in tqdm(eval_files, desc="Processing validation files"):
            result = process_single_file(file_data, lr_output_dir, hr_output_dir, scale_factor, psf_type, sigma, sigma_min, sigma_max, radius, radius_min, radius_max)
            if result:
                eval_results.append(result)
        with open(eval_files_path, "w") as f_eval:
            for result in eval_results:
                f_eval.write(result + "\n")

if __name__ == "__main__":
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Process FITS files with specified PSF and downsampling factor.")
    parser.add_argument("--psf_type", type=str, choices=['gaussian', 'airy'], help="Type of PSF to use (gaussian or airy)")
    parser.add_argument("--sigma", type=float, default=None, help="Fixed sigma for Gaussian PSF")
    parser.add_argument("--sigma_min", type=float, default=0.8, help="Minimum sigma for Gaussian PSF")
    parser.add_argument("--sigma_max", type=float, default=1.2, help="Maximum sigma for Gaussian PSF")
    parser.add_argument("--radius", type=float, default=None, help="Fixed radius for Airy PSF")
    parser.add_argument("--radius_min", type=float, default=1.5, help="Minimum radius for Airy PSF")
    parser.add_argument("--radius_max", type=float, default=1.9, help="Maximum radius for Airy PSF")
    parser.add_argument("--scale_factor", type=int, default=2, help="Downsampling factor (default: 2)")
    parser.add_argument("--datasetlist_path", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/split_file/datasetlist.txt", help="Path to dataset list file")
    parser.add_argument("--lr_output_dir", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/psf_lr", help="Directory to save LR images")
    parser.add_argument("--hr_output_dir", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/psf_hr", help="Directory to save HR images")
    parser.add_argument("--split_file_dir", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/split_file", help="Directory to save split files")
    
    args = parser.parse_args()
    
    # 检查 PSF 参数并提供提示
    if args.psf_type == 'gaussian':
        if args.sigma is not None:
            print(f"Using fixed sigma = {args.sigma} for Gaussian PSF")
        else:
            print(f"Using random sigma between {args.sigma_min} and {args.sigma_max} for Gaussian PSF")
    elif args.psf_type == 'airy':
        if args.radius is not None:
            print(f"Using fixed radius = {args.radius} for Airy PSF")
        else:
            print(f"Using random radius between {args.radius_min} and {args.radius_max} for Airy PSF")
    else:
        print("Warning: PSF type not specified, will randomly choose between Gaussian and Airy.")
    
    # 调用主处理函数
    process_fits_files(
        args.datasetlist_path,
        args.lr_output_dir,
        args.hr_output_dir,
        args.split_file_dir,
        args.scale_factor,
        args.psf_type,
        args.sigma,
        args.sigma_min,
        args.sigma_max,
        args.radius,
        args.radius_min,
        args.radius_max
    )
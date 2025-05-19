import os
import random
import argparse
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve
from astropy.modeling.functional_models import Gaussian2D, AiryDisk2D
from reproject import reproject_exact
import numpy as np
from shapely.wkt import loads
import pdb
def generate_psf(psf_type=None, sigma=None, sigma_min=0.8, sigma_max=1.2, radius=None, radius_min=1.9, radius_max=2.2, size=15):
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

def load_fits(file_path):
    with fits.open(file_path) as hdul:
        image = hdul[1].data.astype(float)
        wcs = WCS(hdul[1].header)
        mean = np.nanmean(image)
        sigma = np.nanstd(image)
        lower_bound = mean - 10 * sigma
        upper_bound = mean + 10 * sigma
        image_clipped = image.copy()
        image_clipped[(image < lower_bound) | (image > upper_bound)] = np.nan
        print(f"HR image mean: {np.nanmean(image_clipped):.6f}, std: {np.nanstd(image_clipped):.6f}, min: {np.nanmin(image_clipped):.6f}, max: {np.nanmax(image_clipped):.6f}")
        print(f"Data type: {image_clipped.dtype}")

        mask = ~np.isnan(image_clipped)
        if np.all(mask):
            mask = np.ones_like(image_clipped, dtype=bool)

        if image_clipped.shape[0] > 6000 or image_clipped.shape[1] > 6000:
            raise ValueError(f"图像 shape {image_clipped.shape} 过大，跳过处理")
        
        return image_clipped, mask, wcs

def apply_psf(image, psf, mask=None):
    if mask is None:
        mask = ~np.isnan(image)
    image_temp = np.where(mask, image, 0.0)
    blurred_image = convolve(image_temp, psf, normalize_kernel=True)
    blurred_image[~mask] = np.nan
    return blurred_image

def apply_poisson_noise(image, mask, scale=100, normalize=False):
    """
    向图像的正数区域添加泊松噪声，负数和 NaN 区域保持不变。
    
    参数:
        image (numpy.ndarray): 输入图像，包含 NaN 区域。
        mask (numpy.ndarray): 布尔掩码，True 表示有效区域。
        scale (float): 缩放因子，控制噪声强度。
        normalize (bool): 是否对图像进行归一化。
    
    返回:
        numpy.ndarray: 添加噪声后的图像，负数和 NaN 区域保持不变。
    """
    # 创建图像副本以保留 NaN 区域
    noisy_image = np.copy(image)
    
    # 提取有效区域的像素值
    valid_image = image[mask]
    
    # 在有效区域内分离正数区域
    positive_mask = valid_image >= 0
    
    # 获取正数区域的值
    positive_values = valid_image[positive_mask]
    
    if normalize:
        # 对正数区域进行归一化
        image_max = np.max(positive_values)
        if image_max > 0:
            image_normalized = positive_values / image_max
        else:
            image_normalized = positive_values
        # 缩放归一化图像
        scaled_image = image_normalized * scale
    else:
        # 直接缩放正数区域
        scaled_image = positive_values * scale
    
    # 添加泊松噪声
    noisy_positive = np.random.poisson(scaled_image)
    
    if normalize:
        # 恢复原始范围
        noisy_positive = (noisy_positive / scale) * image_max
    else:
        noisy_positive = noisy_positive / scale
    
    # 将噪声值填充到正数区域
    positive_indices = np.where(mask & (image >= 0))
    noisy_image[positive_indices] = noisy_positive
    
    # 负数和 NaN 区域已通过副本保留不变
    return noisy_image

def pad_to_multiple(image, mask, wcs, multiple=256, pad_value=np.nan):
    h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple  
    pad_w = (multiple - w % multiple) % multiple  
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_value)
    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)
    padded_wcs = wcs.deepcopy()
    padded_wcs.array_shape = padded_image.shape
    return padded_image, padded_mask, padded_wcs

def downsample_image(image, wcs, scale_factor=2):
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

def save_downsampled_image(image, wcs, output_dir, identifier):
    lr_path = os.path.join(output_dir, f"{identifier}_downsampled.fits")
    hdu = fits.PrimaryHDU(image, header=wcs.to_header())
    hdu.writeto(lr_path, overwrite=True)
    return lr_path

def save_padded_hr_image(image, wcs, output_dir, identifier):
    padded_hr_path = os.path.join(output_dir, f"{identifier}_padded_hr.fits")
    hdu = fits.PrimaryHDU(image, header=wcs.to_header())
    hdu.writeto(padded_hr_path, overwrite=True)
    return padded_hr_path

def process_single_file(file_data, lr_output_dir, hr_output_dir, scale_factor, psf_type, sigma, sigma_min, sigma_max, radius, radius_min, radius_max):
    fits_filepath, padded_image, padded_mask, padded_wcs = file_data
    try:
        identifier = os.path.basename(fits_filepath).replace(".fits", "").replace(".gz", "")
        
        os.makedirs(hr_output_dir, exist_ok=True)
        os.makedirs(lr_output_dir, exist_ok=True)
        
        # First degradation: Gaussian PSF + light Poisson noise
        gaussian_psf = generate_psf('gaussian', sigma, sigma_min, sigma_max)
        blurred_image_gaussian = apply_psf(padded_image, gaussian_psf, mask=padded_mask)
        noisy_image = apply_poisson_noise(blurred_image_gaussian, padded_mask, scale=1000, normalize=True)
        # Downsample after first degradation
        downsampled_image, target_wcs = downsample_image(noisy_image, padded_wcs, scale_factor)
        lr_mask = ~np.isnan(downsampled_image)
        
        # Second degradation: Airy PSF
        airy_psf = generate_psf('airy', radius=radius, radius_min=radius_min, radius_max=radius_max)
        downsampled_image_airy = apply_psf(downsampled_image, airy_psf, mask=lr_mask)
        
        padded_hr_path = save_padded_hr_image(padded_image, padded_wcs, hr_output_dir, identifier)
        lr_path = save_downsampled_image(downsampled_image_airy, target_wcs, lr_output_dir, identifier)
        
        return f"{padded_hr_path},{lr_path}"
    except Exception as e:
        print(f"Processing {fits_filepath} fail: {e}")
        return None

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

        train_results = []
        for file_data in tqdm(train_files, desc="Processing train files"):
            result = process_single_file(file_data, lr_output_dir, hr_output_dir, scale_factor, psf_type, sigma, sigma_min, sigma_max, radius, radius_min, radius_max)
            if result:
                train_results.append(result)
        with open(train_files_path, "w") as f_train:
            for result in train_results:
                f_train.write(result + "\n")
        
        eval_results = []
        for file_data in tqdm(eval_files, desc="Processing validation files"):
            result = process_single_file(file_data, lr_output_dir, hr_output_dir, scale_factor, psf_type, sigma, sigma_min, sigma_max, radius, radius_min, radius_max)
            if result:
                eval_results.append(result)
        with open(eval_files_path, "w") as f_eval:
            for result in eval_results:
                f_eval.write(result + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FITS files with specified PSF and downsampling factor.")
    parser.add_argument("--psf_type", type=str, choices=['gaussian', 'airy'], help="Type of PSF to use (gaussian or airy)")
    parser.add_argument("--sigma", type=float, default=None, help="Fixed sigma for Gaussian PSF")
    parser.add_argument("--sigma_min", type=float, default=0.8, help="Minimum sigma for Gaussian PSF")
    parser.add_argument("--sigma_max", type=float, default=1.2, help="Maximum sigma for Gaussian PSF")
    parser.add_argument("--radius", type=float, default=None, help="Fixed radius for Airy PSF")
    parser.add_argument("--radius_min", type=float, default=1.9, help="Minimum radius for Airy PSF")
    parser.add_argument("--radius_max", type=float, default=2.2, help="Maximum radius for Airy PSF")
    parser.add_argument("--scale_factor", type=int, default=2, help="Downsampling factor (default: 2)")
    parser.add_argument("--datasetlist_path", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_real_world/split_file/datasetlist.txt", help="Path to dataset list file")
    parser.add_argument("--lr_output_dir", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_real_world/psf_lr", help="Directory to save LR images")
    parser.add_argument("--hr_output_dir", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_real_world/psf_hr", help="Directory to save HR images")
    parser.add_argument("--split_file_dir", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_real_world/split_file", help="Directory to save split files")
    
    args = parser.parse_args()
    
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
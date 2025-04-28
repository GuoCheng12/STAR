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
from astropy.table import Table
import json
import pdb
from photutils.aperture import CircularAperture, aperture_photometry

def generate_psf(psf_type=None, sigma=None, sigma_min=0.8, sigma_max=1.2, radius=None, radius_min=1.5, radius_max=1.9, size=15):
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

def perform_psf_photometry(image, mask):
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

def perform_psf_photometry_lr(lr_image, lr_mask, hr_stars, hr_wcs, lr_wcs, scale_factor=2):
    """
    在 LR 图像上进行 PSF 测光，使用 HR 图像的恒星位置（通过 WCS 映射到 LR 坐标系），只优化 flux。
    只保留在 LR 中成功拟合的恒星，并返回匹配的 HR 恒星子集。
    
    参数:
    - lr_image: LR 图像数据 (numpy array)
    - lr_mask: LR 图像的掩码 (boolean array, True 表示有效像素)
    - hr_stars: HR 图像中检测到的恒星列表，包含 'x'、'y' 和 'flux'
    - hr_wcs: HR 图像的 WCS 对象 (astropy.wcs.WCS)
    - lr_wcs: LR 图像的 WCS 对象 (astropy.wcs.WCS)
    - scale_factor: 下采样因子 (int, 默认值为 2，仅用于参数调整)
    
    返回:
    - lr_stars: LR 图像中测光得到的恒星列表，包含 'x'、'y' 和 'flux'
    - matched_hr_stars: 与 LR 恒星匹配的 HR 恒星子集
    """
    bkg = Background2D(lr_image, (64 // scale_factor, 64 // scale_factor), filter_size=(3, 3), 
                       bkg_estimator=MedianBackground(), mask=~lr_mask)
    data_sub = lr_image - bkg.background
    data_sub_masked = np.where(lr_mask, data_sub, np.nan)
    
    positions = []
    for star in hr_stars:
        ra, dec = hr_wcs.pixel_to_world_values(star['x'], star['y'])
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
        if 0 <= y < lr_mask.shape[0] and 0 <= x < lr_mask.shape[1] and lr_mask[int(np.round(y)), int(np.round(x))]:
            lr_stars.append({'x': x, 'y': y, 'flux': float(row['flux_fit'])})
            matched_hr_stars.append(hr_stars[i])

    filtered_lr_stars = len(init_params) - len(lr_stars)
    print(f"Number of LR stars filtered out: {filtered_lr_stars}")
    return lr_stars, matched_hr_stars

def apply_psf(image, psf, mask=None):
    if mask is None:
        mask = ~np.isnan(image)
    image_temp = np.where(mask, image, 0.0)
    blurred_image = convolve(image_temp, psf, normalize_kernel=True)
    blurred_image[~mask] = np.nan
    return blurred_image

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

        hr_stars = perform_psf_photometry(padded_image, padded_mask)
        
        psf = generate_psf(psf_type, sigma, sigma_min, sigma_max, radius, radius_min, radius_max)
        blurred_image = apply_psf(padded_image, psf, mask=padded_mask)
        downsampled_image, target_wcs = downsample_image(blurred_image, padded_wcs, scale_factor)
        lr_mask = ~np.isnan(downsampled_image)
        
        lr_stars, matched_hr_stars = perform_psf_photometry_lr(downsampled_image, lr_mask, hr_stars, padded_wcs, target_wcs, scale_factor)
        
        stars_file = os.path.join(hr_output_dir, f"{identifier}_stars.json")
        with open(stars_file, 'w') as f:
            json.dump(matched_hr_stars, f, indent=4)
        
        lr_stars_file = os.path.join(lr_output_dir, f"{identifier}_lr_stars.json")
        if not lr_stars:
            print(f"Warning: No stars detected in LR image for {identifier}")
        else:
            with open(lr_stars_file, 'w') as f:
                json.dump(lr_stars, f, indent=4)
        
        padded_hr_path = save_padded_hr_image(padded_image, padded_wcs, hr_output_dir, identifier)
        lr_path = save_downsampled_image(downsampled_image, target_wcs, lr_output_dir, identifier)
        print(f"NaN of HR stars: {len(hr_stars) - len(matched_hr_stars)}")
        print(f"Number of matched HR stars: {len(matched_hr_stars)}")
        print(f"Number of LR stars: {len(lr_stars)}")
        
        return f"{padded_hr_path},{lr_path},{stars_file},{lr_stars_file}"
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
    parser.add_argument("--radius_min", type=float, default=1.5, help="Minimum radius for Airy PSF")
    parser.add_argument("--radius_max", type=float, default=1.9, help="Maximum radius for Airy PSF")
    parser.add_argument("--scale_factor", type=int, default=2, help="Downsampling factor (default: 2)")
    parser.add_argument("--datasetlist_path", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/split_file/datasetlist.txt", help="Path to dataset list file")
    parser.add_argument("--lr_output_dir", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/psf_lr", help="Directory to save LR images")
    parser.add_argument("--hr_output_dir", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/psf_hr", help="Directory to save HR images")
    parser.add_argument("--split_file_dir", type=str, default="/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy_new/split_file", help="Directory to save split files")
    
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
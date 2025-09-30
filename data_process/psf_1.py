import os
import random
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve
from astropy.modeling.functional_models import Gaussian2D, AiryDisk2D
from reproject import reproject_exact
import numpy as np
from shapely.wkt import loads

# Generate random PSF
def generate_random_psf(size=15, sigma_range=[0.8, 1.2], radius_range=[0.5, 2.0]):
    "Generate a random PSF, randomly select from Gaussian and Airy, and print the parameters."
    psf_type = random.choice(['gaussian'])
    if psf_type == 'gaussian':
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        half_size = size // 2
        x = np.arange(-half_size, half_size + 1)
        y = np.arange(-half_size, half_size + 1)
        x, y = np.meshgrid(x, y)
        psf = Gaussian2D(amplitude=1.0, x_mean=0, y_mean=0, x_stddev=sigma, y_stddev=sigma)(x, y)
        print(f"Generated Gaussian PSF with sigma = {sigma}")
    elif psf_type == 'airy':
        radius = random.uniform(radius_range[0], radius_range[1])
        half_size = size // 2
        x = np.arange(-half_size, half_size + 1)
        y = np.arange(-half_size, half_size + 1)
        x, y = np.meshgrid(x, y)
        psf = AiryDisk2D(amplitude=1.0, x_0=0, y_0=0, radius=radius)(x, y)
        print(f"Generated Airy PSF with radius = {radius}")
    return psf / psf.sum()

# Load FITS file and apply 3-sigma clipping
def load_fits(file_path):
    "Load FITS file, apply 3-sigma clipping to remove noise, return image data, mask, and WCS information"
    with fits.open(file_path) as hdul:
        image = hdul[1].data.astype(float)  
        wcs = WCS(hdul[1].header)
        # 3-sigma clipping removes noise
        mean = np.nanmean(image) 
        sigma = np.nanstd(image)
        lower_bound = mean - 10 * sigma
        upper_bound = mean + 10 * sigma
        image_clipped = image.copy()
        image_clipped[(image < lower_bound) | (image > upper_bound)] = np.nan

        mask = ~np.isnan(image_clipped)
        if np.all(mask):
            mask = np.ones_like(image_clipped, dtype=bool)

        if image_clipped.shape[0] > 6000 or image_clipped.shape[1] > 6000:
            raise ValueError(f"image shape {image_clipped.shape} oversize, skip it")
        
        return image_clipped, mask, wcs

# apply PSF blur
def apply_psf(image, psf, mask=None):
    """Apply PSF blur, use the image's default mask when mask is None."""
    if mask is None:
        mask = ~np.isnan(image)
    image_temp = np.where(mask, image, 0.0)
    blurred_image = convolve(image_temp, psf, normalize_kernel=True)
    blurred_image[~mask] = np.nan
    return blurred_image

# Padding 
def pad_to_multiple(image, mask, wcs, multiple=256, pad_value=np.nan):
    """Pad the image and mask to the specified multiple and update the WCS"""
    h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple  
    pad_w = (multiple - w % multiple) % multiple  
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_value)
    padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)
    padded_wcs = wcs.deepcopy()
    padded_wcs.array_shape = padded_image.shape
    return padded_image, padded_mask, padded_wcs

# Downsampling
def downsample_image(image, wcs, scale_factor=2):
    """Downsample the image and update the WCS"""
    target_shape = (int(image.shape[0] / scale_factor), int(image.shape[1] / scale_factor))
    target_wcs = wcs.deepcopy()
    target_wcs.wcs.crpix = [crpix / scale_factor for crpix in wcs.wcs.crpix]
    if hasattr(wcs.wcs, 'cd'):
        target_wcs.wcs.cd = wcs.wcs.cd * scale_factor
    else:
        target_wcs.wcs.cdelt = [cdelt * scale_factor for cdelt in wcs.wcs.cdelt]
    downsampled_image, _ = reproject_exact((image, wcs), target_wcs, shape_out=target_shape)
    downsampled_image = scale_factor**2 * downsampled_image 
    return downsampled_image, target_wcs

# Save downsampled image
def save_downsampled_image(image, wcs, output_dir, identifier):
    """Save the downsampled image as a FITS file"""
    lr_path = os.path.join(output_dir, f"{identifier}_downsampled.fits")
    hdu = fits.PrimaryHDU(image, header=wcs.to_header())
    hdu.writeto(lr_path, overwrite=True)
    return lr_path

# Save HR image after padding
def save_padded_hr_image(image, wcs, output_dir, identifier):
    """Save the HR image after padding as a FITS file"""
    padded_hr_path = os.path.join(output_dir, f"{identifier}_padded_hr.fits")
    hdu = fits.PrimaryHDU(image, header=wcs.to_header())
    hdu.writeto(padded_hr_path, overwrite=True)
    return padded_hr_path

# Single file processing function
def process_single_file(file_data, lr_output_dir, hr_output_dir, scale_factor=2):
    """Process a single FITS file, including PSF blurring, downsampling, and saving"""
    fits_filepath, padded_image, padded_mask, padded_wcs = file_data
    try:
        identifier = os.path.basename(fits_filepath).replace(".fits", "").replace(".gz", "")
        padded_hr_path = save_padded_hr_image(padded_image, padded_wcs, hr_output_dir, identifier)
        psf = generate_random_psf()
        blurred_image = apply_psf(padded_image, psf, mask=padded_mask)
        downsampled_image, target_wcs = downsample_image(blurred_image, padded_wcs, scale_factor)
        lr_path = save_downsampled_image(downsampled_image, target_wcs, lr_output_dir, identifier)
        return f"{padded_hr_path},{lr_path}"
    except Exception as e:
        print(f"Processing {fits_filepath} fail: {e}")
        return None

# main function
def process_fits_files(datasetlist_path, lr_output_dir, hr_output_dir, split_file_dir, scale_factor=2):
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
                    # if ncombine == 4:
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
            result = process_single_file(file_data, lr_output_dir, hr_output_dir, scale_factor)
            if result:
                train_results.append(result)
        with open(train_files_path, "w") as f_train:
            for result in train_results:
                f_train.write(result + "\n")
        eval_results = []
        for file_data in tqdm(eval_files, desc="Processing validation files"):
            result = process_single_file(file_data, lr_output_dir, hr_output_dir, scale_factor)
            if result:
                eval_results.append(result)
        with open(eval_files_path, "w") as f_eval:
            for result in eval_results:
                f_eval.write(result + "\n")

if __name__ == "__main__":
    datasetlist_path = "dataset/x2/datalist.txt/"
    lr_output_dir = "dataset/x2/psf_lr"
    hr_output_dir = "dataset/x2/psf_hr"
    split_file_dir = "dataset/x2/data_process/split_file"
    process_fits_files(datasetlist_path, lr_output_dir, hr_output_dir, split_file_dir)
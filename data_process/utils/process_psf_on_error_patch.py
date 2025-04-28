import os
import glob
import numpy as np
import sep
from astropy.stats import sigma_clipped_stats
from scipy.stats import multivariate_normal
from tqdm import tqdm
import pdb
# Read error file paths from error_files.txt
path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy/eval_hr_patch"
error_files_path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy/eval_hr_patch/error_files_eval.txt"
with open(error_files_path, 'r') as f:
    npy_files = [line.strip() for line in f.readlines()]

# List to store files that fail during processing
error_files = []

for npy_file in tqdm(npy_files, desc="Processing error .npy files"):
    try:
        name = npy_file.split('/')[-1]
        data = np.load(npy_file, allow_pickle=True).item()
        image = data['image'].astype(np.float32)
        mask = data['mask']  # True indicates valid regions
        image_cleaned = np.where(mask, image, 0.0)
        bkg = sep.Background(image_cleaned, mask=~mask, bw=64, bh=64, fw=1, fh=1)
        image_sub = image_cleaned - bkg.back()
        valid_pixels = image_sub[mask]
        mean, median, std = sigma_clipped_stats(image_sub[mask], sigma=3.0)
        sources = sep.extract(image_sub, 1.5, err=bkg.rms(), mask=~mask)

        theta = sources['theta']
        theta = np.mod(theta + np.pi/2, np.pi) - np.pi/2
        print(f"{name} 检测到 {len(sources)} 个源。")

        try:
            flux, fluxerr, flag = sep.sum_ellipse(image_sub, sources['x'], sources['y'],
                                                  sources['a'], sources['b'], theta,
                                                  1.5, err=bkg.globalrms)
            valid_idx = ~np.isnan(flux)
            sources = sources[valid_idx]
            flux_cleaned = flux[valid_idx]
        except Exception as e:
            print(f"测光失败: {e}")
            flux_cleaned = np.array([])  # Empty flux on photometry failure
            error_files.append(npy_file)  # Log the failed file

        # Step 5: Generate attn_map
        def generate_attn_map(image_shape, sources, flux):
            attn_map = np.zeros(image_shape, dtype=np.float32)
            max_flux = np.max(flux) if len(flux) > 0 else 1.0  # Prevent division by zero

            for i in range(len(sources)):
                x, y = sources['x'][i], sources['y'][i]
                a, b = sources['a'][i], sources['b'][i]
                theta = sources['theta'][i]
                sigma_x = a
                sigma_y = b
                cov = np.array([[sigma_x**2, 0], [0, sigma_y**2]])
                rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                       [np.sin(theta), np.cos(theta)]])
                cov_rot = rot_matrix @ cov @ rot_matrix.T
                gauss = multivariate_normal(mean=[x, y], cov=cov_rot)
                x_grid, y_grid = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
                pos = np.dstack((x_grid, y_grid))
                weights = gauss.pdf(pos)

                weights /= weights.max() if weights.max() > 0 else 1.0  # Normalize
                weights *= flux[i] / max_flux  # Weight by flux
                attn_map += weights

            return attn_map

        if len(sources) > 0 and len(flux_cleaned) > 0:
            attn_map = generate_attn_map(image_sub.shape, sources, flux_cleaned)
        else:
            # Generate zero attn_map if no valid sources or photometry fails
            attn_map = np.zeros_like(image_sub)

        # Apply mask
        attn_map = np.where(mask, attn_map, np.nan)

        # Save attn_map to the original file
        data['attn_map'] = attn_map
        np.save(npy_file, data)
    except Exception as e:
        print(f"处理 {npy_file} 时出错: {e}")
        error_files.append(npy_file)  # Log processing failures

# Save failed files to a new text file
error_txt_path = os.path.join(path, "error_files_new.txt")
with open(error_txt_path, "w") as f:
    for error_file in error_files:
        f.write(error_file + "\n")
print(f"异常文件已保存到: {error_txt_path}")
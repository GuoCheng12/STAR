import os
import glob
import numpy as np
import sep
from astropy.stats import sigma_clipped_stats
from scipy.stats import multivariate_normal
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random
def process_npy_file(npy_file):
    """Function to process a single .npy file"""
    try:
        name = npy_file.split('/')[-1]
        data = np.load(npy_file, allow_pickle=True).item()
        image = data['image'].astype(np.float32)
        mask = data['mask']  
        image_cleaned = np.where(mask, image, 0.0)

        bkg = sep.Background(image_cleaned, mask=~mask, bw=32, bh=32, fw=1, fh=1)
        image_sub = image_cleaned - bkg.back()
        sources = sep.extract(image_sub, 1.5, err=bkg.rms(), mask=~mask)
        print(f"{name} Detected {len(sources)} sources。")

        try:
            flux, fluxerr, flag = sep.sum_ellipse(image_sub, sources['x'], sources['y'],
                                                  sources['a'], sources['b'], sources['theta'],
                                                  2.5, err=bkg.globalrms)
            valid_idx = ~np.isnan(flux)
            sources = sources[valid_idx]
            flux_cleaned = flux[valid_idx]
        except Exception as e:
            print(f"photometric fail (fw=1, fh=1): {e}")
            bkg = sep.Background(image_cleaned, mask=~mask, bw=32, bh=32, fw=3, fh=3)
            image_sub = image_cleaned - bkg.back()
            sources = sep.extract(image_sub, 1.5, err=bkg.rms(), mask=~mask)
            try:
                flux, fluxerr, flag = sep.sum_ellipse(image_sub, sources['x'], sources['y'],
                                                      sources['a'], sources['b'], sources['theta'],
                                                      2.5, err=bkg.globalrms)
                valid_idx = ~np.isnan(flux)
                sources = sources[valid_idx]
                flux_cleaned = flux[valid_idx]
            except Exception as e:
                print(f"photometric fail (fw=3, fh=3): {e}")
                flux_cleaned = np.array([])  
        from scipy.special import j1  

        def airy_kernel(radius_pixels, size=51):
            """Generate a centrally symmetric Airy disk kernel (two-dimensional)."""
            assert size % 2 == 1, "Size must be odd"
            center = size // 2
            y, x = np.ogrid[:size, :size]
            r = np.sqrt((x - center)**2 + (y - center)**2)
            kr = np.pi * r / radius_pixels + 1e-8 
            airy = (2 * j1(kr) / kr)**2
            airy /= airy.max()
            return airy

        def generate_attn_map_airy(image_shape, sources, flux, pixel_scale=1.0):
            """Based on Airy disk attention map generation"""
            attn_map = np.zeros(image_shape, dtype=np.float32)

            for i in range(len(sources)):
                x, y = sources['x'][i], sources['y'][i]
                abs_flux = np.abs(flux[i])

                radius = max(sources['a'][i], sources['b'][i]) 
                kernel_size = int(6 * radius)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                airy_mask = airy_kernel(radius, size=kernel_size)
                xmin = int(max(0, x - kernel_size // 2))
                xmax = int(min(image_shape[1], x + kernel_size // 2 + 1))
                ymin = int(max(0, y - kernel_size // 2))
                ymax = int(min(image_shape[0], y + kernel_size // 2 + 1))

                kx1 = max(0, kernel_size // 2 - int(x - xmin))
                ky1 = max(0, kernel_size // 2 - int(y - ymin))
                kx2 = kx1 + (xmax - xmin)
                ky2 = ky1 + (ymax - ymin)

                attn_map[ymin:ymax, xmin:xmax] += airy_mask[ky1:ky2, kx1:kx2] * abs_flux

            return attn_map
        def generate_attn_map(image_shape, sources, flux):
            """Generate attention map"""
            attn_map = np.zeros(image_shape, dtype=np.float32)
            for i in range(len(sources)):
                x, y = sources['x'][i], sources['y'][i]
                a, b = sources['a'][i], sources['b'][i]
                theta = sources['theta'][i]
                cov = np.array([[a**2, 0], [0, b**2]])
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                rot_matrix = np.array([[cos_theta, -sin_theta], 
                                       [sin_theta, cos_theta]])
                cov_rot = rot_matrix @ cov @ rot_matrix.T
                gauss = multivariate_normal(mean=[x, y], cov=cov_rot)
                x_grid, y_grid = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
                pos = np.dstack((x_grid, y_grid))
                weights = gauss.pdf(pos)
                abs_flux = np.abs(flux[i])
                attn_map += weights * abs_flux
            return attn_map

        if len(sources) > 0 and len(flux_cleaned) > 0:
            f=open('train2.txt','a')
            f.write(str(len(sources))+',')
            num = random.randint(0, 1)
            if num>0.5:
                attn_map = generate_attn_map_airy(image_sub.shape, sources, flux_cleaned)
            else:
                attn_map = generate_attn_map(image_sub.shape, sources, flux_cleaned)
        else:
            attn_map = np.zeros_like(image_sub) 
        attn_map = np.where(mask, attn_map, np.nan)

        data['attn_map'] = attn_map
        np.save(npy_file, data)
    except Exception as e:
        print(f"processing file {npy_file} wrong: {e}")
        return npy_file  
    return None

if __name__ == "__main__":

    path = "dataset/x4/eval_hr_patch"
    npy_files = glob.glob(os.path.join(path, "*.npy"))
    error_files = []
    
    # Use a multiprocessing pool and display a progress bar
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_npy_file, npy_files), total=len(npy_files), 
                            desc="Processing .npy files"))
        

 
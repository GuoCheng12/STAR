import os
import glob
import numpy as np
import sep
from astropy.stats import sigma_clipped_stats
from scipy.stats import multivariate_normal
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_npy_file(npy_file):
    """处理单个 .npy 文件的函数"""
    try:
        name = npy_file.split('/')[-1]
        data = np.load(npy_file, allow_pickle=True).item()
        image = data['image'].astype(np.float32)
        mask = data['mask']  # True 表示有效区域
        image_cleaned = np.where(mask, image, 0.0)

        # 第一次尝试背景减除 (fw=1, fh=1)
        bkg = sep.Background(image_cleaned, mask=~mask, bw=64, bh=64, fw=1, fh=1)
        image_sub = image_cleaned - bkg.back()
        sources = sep.extract(image_sub, 1.5, err=bkg.rms(), mask=~mask)
        print(f"{name} 检测到 {len(sources)} 个源。")

        try:
            flux, fluxerr, flag = sep.sum_ellipse(image_sub, sources['x'], sources['y'],
                                                  sources['a'], sources['b'], sources['theta'],
                                                  2.5, err=bkg.globalrms)
            valid_idx = ~np.isnan(flux)
            sources = sources[valid_idx]
            flux_cleaned = flux[valid_idx]
        except Exception as e:
            print(f"测光失败 (fw=1, fh=1): {e}")
            # 第二次尝试背景减除 (fw=3, fh=3)
            bkg = sep.Background(image_cleaned, mask=~mask, bw=64, bh=64, fw=3, fh=3)
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
                print(f"测光失败 (fw=3, fh=3): {e}")
                flux_cleaned = np.array([])  # 测光失败时，设为空流量

        def generate_attn_map(image_shape, sources, flux):
            """生成注意力图"""
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
            attn_map = generate_attn_map(image_sub.shape, sources, flux_cleaned)
        else:
            attn_map = np.zeros_like(image_sub)  # 无有效源或测光失败，设为全 0
        attn_map = np.where(mask, attn_map, np.nan)

        data['attn_map'] = attn_map
        np.save(npy_file, data)
    except Exception as e:
        print(f"处理 {npy_file} 时出错: {e}")
        np.save(npy_file, data)
        return npy_file  # 返回错误文件路径
    return None

if __name__ == "__main__":
    path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_real_world/eval_hr_patch"
    npy_files = glob.glob(os.path.join(path, "*.npy"))
    error_files = []

    # 使用多进程池并显示进度条
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_npy_file, npy_files), total=len(npy_files), 
                            desc="Processing .npy files"))
    
    # 收集处理失败的文件
    error_files = [result for result in results if result is not None]

    # 保存错误文件路径
    error_txt_path = os.path.join(path, "error_files.txt")
    with open(error_txt_path, "w") as f:
        for error_file in error_files:
            f.write(error_file + "\n")
    print(f"异常文件已保存到: {error_txt_path}")
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, Gaussian2DKernel
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.psf import CircularGaussianPRF, PSFPhotometry
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
from photutils.psf import MoffatPSF
from astropy.modeling.models import Moffat2D
from photutils.psf import PSFPhotometry, FittableImageModel
def generate_lr_image(hr_image, psf_sigma=1.5, scale_factor=2):
    """生成低分辨率图像（PSF模糊 + 下采样）"""
    # 生成高斯PSF核
    kernel = Gaussian2DKernel(x_stddev=psf_sigma)
    kernel.normalize(mode='integral')
    
    # 应用PSF模糊
    blurred = convolve(hr_image, kernel, boundary='extend')
    
    # 下采样
    lr_image = blurred[::scale_factor, ::scale_factor] 
    lr_image *= scale_factor**2  # 保持通量
    
    return lr_image


def perform_psf_photometry(image, mask):
    """在图像上进行PSF测光，返回恒星的坐标和通量信息，并验证StarFinder与PSF拟合结果的差异。"""
    # 计算背景并减去
    bkg = Background2D(image, (64, 64), filter_size=(3, 3), 
                       bkg_estimator=MedianBackground(), mask=~mask)
    data_sub = image - bkg.background
    data_sub_masked = np.where(mask, data_sub, np.nan)


    # 统计图像参数
    mean, median, std = sigma_clipped_stats(data_sub_masked, sigma=3.0)
    
    # 使用 DAOStarFinder 检测恒星
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

    # 验证代码：比较 DAOStarFinder 和 PSFPhotometry 的结果
    print("比较 DAOStarFinder 和 PSFPhotometry 的结果：")
    for i in range(len(sources_tbl)):
        if i < len(phot_table):  # 确保 PSFPhotometry 成功拟合
            x_init = sources_tbl['x_0'][i]
            y_init = sources_tbl['y_0'][i]
            flux_init = sources_tbl['flux_0'][i]
            x_fit = phot_table['x_fit'][i]
            y_fit = phot_table['y_fit'][i]
            flux_fit = phot_table['flux_fit'][i]
            dx = x_fit - x_init
            dy = y_fit - y_init
            dflux = flux_fit - flux_init
            print(f"Star {i}:")
            print(f"  Initial position: x={x_init:.2f}, y={y_init:.2f}")
            print(f"  Fitted position: x={x_fit:.2f}, y={y_fit:.2f}")
            print(f"  Position difference: dx={dx:.2f}, dy={dy:.2f}")
            print(f"  Initial flux: {flux_init:.2f}")
            print(f"  Fitted flux: {flux_fit:.2f}")
            print(f"  Flux difference: {dflux:.2f}")
        else:
            print(f"Star {i}: PSFPhotometry 拟合失败")

    return stars

def patchify_hr(image, mask, stars, patch_size=256, stride=128, useful_region_th=0.8):
    """HR分块处理"""
    hr_patches = []
    h, w = image.shape
    recorded_stars = set()
    tolerance = 1e-6
    # 生成HR分块
    for x_start in range(0, h - patch_size + 1, stride):
        for y_start in range(0, w - patch_size + 1, stride):
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            mask_patch = mask[x_start:x_end, y_start:y_end]

            if mask_patch.mean() > useful_region_th:
                hr_stars_in_patch = []
                for star in stars:
                    star_col = star['x']
                    star_row = star['y']
                    # 使用容差判断并检查原始mask
                    if (y_start - tolerance <= star_col < y_end + tolerance and 
                        x_start - tolerance <= star_row < x_end + tolerance and
                        mask[int(star_row), int(star_col)]):  # 直接截断坐标
                        hr_stars_in_patch.append({
                            'rel_x': star_col - y_start,
                            'rel_y': star_row - x_start,
                            'flux': star['flux']
                        })
                        recorded_stars.add((star_col, star_row))
                hr_patches.append((
                    image[x_start:x_end, y_start:y_end],
                    mask_patch,
                    (x_start, y_start),  # (行起始，列起始) 左上角
                    hr_stars_in_patch
                ))

    # 验证恒星分配
    all_stars = {(s['x'], s['y']) for s in stars}
    missing = all_stars - recorded_stars
    if missing:
        print(f"HR分块遗漏恒星: {len(missing)}")

    return hr_patches

def patchify_lr(lr_image, lr_mask, hr_patches, scale_factor=2, lr_patch_size=128):
    """基于HR分块坐标生成LR分块"""
    lr_patches = []
    import pdb; pdb.set_trace()
    for hr_patch in hr_patches:
        hr_row_start, hr_col_start = hr_patch[2]  # 获取HR分块起始坐标
        
        # 计算对应的LR分块坐标
        lr_row_start = hr_row_start // scale_factor
        lr_col_start = hr_col_start // scale_factor
        lr_row_end = lr_row_start + lr_patch_size
        lr_col_end = lr_col_start + lr_patch_size
        
        # 边界检查
        lr_row_end = min(lr_row_end, lr_image.shape[0])
        lr_col_end = min(lr_col_end, lr_image.shape[1])
        
        # 提取LR分块
        lr_patch = lr_image[lr_row_start:lr_row_end, lr_col_start:lr_col_end]
        lr_mask_patch = lr_mask[lr_row_start:lr_row_end, lr_col_start:lr_col_end]
        
        # 收集属于当前LR分块的恒星
        lr_stars = []
        for star in hr_patch[3]:  # 遍历对应HR分块中的恒星
            # 转换到LR坐标系
            lr_rel_x = star['rel_x'] / scale_factor
            lr_rel_y = star['rel_y'] / scale_factor
            
            # 检查是否在LR分块范围内
            if (0 <= lr_rel_x < lr_patch_size) and (0 <= lr_rel_y < lr_patch_size):
                lr_stars.append({
                    'rel_x': lr_rel_x,
                    'rel_y': lr_rel_y,
                    'flux': star['flux']
                })
        
        lr_patches.append((
            lr_patch,
            lr_mask_patch,
            (lr_row_start, lr_col_start),
            lr_stars
        ))
    
    return lr_patches



def plot_combined_patches_with_stars(hr_image, lr_image, hr_patches, lr_patches, scale_factor=2):
    """可视化HR/LR分块及恒星位置对应关系"""
    plt.figure(figsize=(20, 10))
    
    # --------------------------
    # HR图像可视化
    # --------------------------
    plt.subplot(121)
    plt.imshow(hr_image, cmap='gray', origin='lower', vmax=np.nanpercentile(hr_image, 99))
    plt.title('High-Resolution (HR) with Stars')
    
    # 绘制HR分块框和恒星
    for patch in hr_patches:
        hr_patch, hr_mask, (row_start, col_start), hr_stars = patch
        
        # 绘制分块框
        rect = patches.Rectangle(
            (col_start, row_start), 256, 256,
            linewidth=1, edgecolor='red', facecolor='none'
        )
        plt.gca().add_patch(rect)
        
        # 绘制HR恒星（绝对坐标）
        for star in hr_stars:
            abs_x = col_start + star['rel_x']
            abs_y = row_start + star['rel_y']
            plt.plot(abs_x, abs_y, 'ro', markersize=4, alpha=0.7)
    
    # --------------------------
    # LR图像可视化（叠加到HR坐标系）
    # --------------------------
    plt.subplot(122)
    plt.imshow(lr_image, cmap='gray', origin='lower', 
             extent=[0, hr_image.shape[1], 0, hr_image.shape[0]],  # 保持HR坐标系
             vmax=np.nanpercentile(lr_image, 99))
    plt.title('Low-Resolution (LR) Overlay with Stars')
    
    # 绘制LR分块框和恒星
    for lr_patch in lr_patches:
        lr_img, lr_mask, (lr_row, lr_col), lr_stars = lr_patch
        
        # 转换到HR坐标系
        hr_col_start = lr_col * scale_factor
        hr_row_start = lr_row * scale_factor
        
        # 绘制LR分块框（虚线框）
        rect = patches.Rectangle(
            (hr_col_start, hr_row_start), 
            128*scale_factor, 128*scale_factor,
            linewidth=1, edgecolor='blue', linestyle='--', facecolor='none'
        )
        plt.gca().add_patch(rect)
        
        # 绘制LR恒星（转换到HR坐标系）
        for star in lr_stars:
            lr_abs_x = lr_col + star['rel_x']
            lr_abs_y = lr_row + star['rel_y']
            hr_abs_x = lr_abs_x * scale_factor
            hr_abs_y = lr_abs_y * scale_factor
            plt.plot(hr_abs_x, hr_abs_y, 'b*', markersize=8, alpha=0.7)
    
    # --------------------------
    # 图例和保存
    # --------------------------
    plt.legend([patches.Patch(color='red', label='HR Patches'),
                patches.Patch(color='blue', label='LR Patches'),
                plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=4, label='HR Stars'),
                plt.Line2D([], [], color='blue', marker='*', linestyle='None', markersize=8, label='LR Stars')],
               ['HR Patches', 'LR Patches', 'HR Stars', 'LR Stars'])
    
    plt.savefig('HR_LR_patch_star_alignment.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """主函数，执行FITS图像测光和分割"""
    # 读取FITS图像
    fits_file = "/home/bingxing2/ailab/group/ai4astro/Datasets/zgh/origin_fits/hst_10152_05_acs_wfc_f814w_j90i05_drc.fits.gz"
    with fits.open(fits_file) as hdul:
        hr_image = hdul[1].data
    
    # 2. 生成mask和padding
    mask = ~np.isnan(hr_image)
    hr_padded = np.pad(hr_image, ((0, 256-hr_image.shape[0]%256), 
                                 (0, 256-hr_image.shape[1]%256)), 
                      mode='constant', constant_values=np.nan)
    mask_padded = np.pad(mask, ((0, 256-mask.shape[0]%256), 
                               (0, 256-mask.shape[1]%256)), 
                       mode='constant', constant_values=False)
    
    # 3. PSF测光
    stars = perform_psf_photometry(hr_padded, mask_padded)
    
    # 4. 生成LR图像
    lr_image = generate_lr_image(hr_padded)
    lr_mask = ~np.isnan(lr_image)
    
    # 5. 分块处理
    hr_patches = patchify_hr(hr_padded, mask_padded, stars)
    lr_patches = patchify_lr(lr_image, lr_mask, hr_patches)
    plot_combined_patches_with_stars(hr_padded, lr_image, hr_patches, lr_patches)
    # # 6. 可视化验证
    # plot_combined_patches(hr_padded, lr_image, hr_patches, lr_patches)
    
    # # 7. 保存示例数据
    # np.save('hr_patch_example.npy', hr_patches[0][0])
    # np.save('lr_patch_example.npy', lr_patches[0][0])
    
    # # 可视化结果
    # plot_original_with_stars(image_padded, stars)          # padding后的图像 + 恒星
    # plot_patches_with_stars(image_padded, patches)         # padding后的图像 + patch边框 + 恒星

if __name__ == "__main__":
    main()
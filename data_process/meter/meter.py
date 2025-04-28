import numpy as np
from imageio import imread
from astropy.stats import sigma_clipped_stats
from photutils.aperture import aperture_photometry, CircularAperture
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.psf import PSFPhotometry, MoffatPSF, SourceGrouper
from astropy.io import fits
from astropy.table import QTable
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sep
from matplotlib.patches import Ellipse
import pandas as pd
import pdb
import time
#====================================================
# 1) 读取FITS图像并创建掩码
#====================================================

# 读取 FITS 文件
filename = "/home/bingxing2/ailab/group/ai4astro/Datasets/zgh/origin_fits/hst_10152_05_acs_wfc_f814w_j90i05_drc.fits.gz"
hdul = fits.open(filename)
data = hdul[1].data.astype(np.float32)  # 转换为 float32 以兼容 sep
mask = ~np.isnan(data)  # 创建掩码，True 表示有效区域
hdul.close()
sep.set_sub_object_limit(8192)
# 背景扣除
box_size = 64
bkg = sep.Background(data, mask=mask, bw=box_size, bh=box_size, fw=3, fh=3)
data_sub = data - bkg.back()

# # 将无效区域掩码为 NaN
# data_sub_masked = np.where(mask, data_sub, np.nan)

# # 计算图像统计量
mean_val, median_val, std_val = sigma_clipped_stats(data_sub, sigma=3.0)

# # 设置检测阈值
threshold = 15 * std_val  # 检测阈值：5σ，可调整
start_time = time.time()
# 使用 SExtractor (sep) 进行源检测
sources = sep.extract(data_sub, threshold, err=bkg.globalrms)
end_time = time.time()

# 检查检测结果
if len(sources) == 0:
    print("未检测到源。请调整 'threshold'。")
else:
    print(f"检测到 {len(sources)} 个源。")


# Z-scale 归一化
zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(data_sub)

# 可视化
fig, ax = plt.subplots(figsize=(12, 12), dpi=300)  # 提高分辨率
im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
               vmin=vmin, vmax=vmax, origin='lower')

# 绘制椭圆标识每个源
for i in range(len(sources)):
    e = Ellipse(xy=(sources['x'][i], sources['y'][i]),
                width=6*sources['a'][i],
                height=6*sources['b'][i],
                angle=sources['theta'][i] * 180. / np.pi)
    e.set_facecolor('none')
    e.set_edgecolor('red')
    ax.add_artist(e)

plt.tight_layout()
plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/sextractor_detection.png')
#====================================================
# 4) 孔径测光 (Aperture Photometry)
#====================================================

if sources is not None and len(sources) > 0:
    positions = np.transpose((sources['xcentroid'].values, sources['ycentroid'].values))
    aperture_radius = 2.0  # 孔径半径（像素）

    apertures = CircularAperture(positions, r=aperture_radius)
    phot_table = aperture_photometry(data_sub, apertures, mask=~mask)  # ~mask: True 表示无效区域

    # 提取孔径测光结果
    flux_aperture = phot_table['aperture_sum']

    print("孔径测光结果：")
    print("检测到的源数量：", len(flux_aperture))
    print("总通量：", flux_aperture.sum())

    # 可视化检测到的源，用红色圆圈表示
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.imshow(data_sub, origin='lower', cmap='gray', norm=norm_sub)
    ax.set_title('Aperture Photometry (red circles)')
    for x, y in positions:
        circ = Circle((x, y), radius=aperture_radius, edgecolor='red', facecolor='none', lw=1)
        ax.add_patch(circ)
    plt.tight_layout()
    plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/aperture_photometry.png')
    pdb.set_trace()
#====================================================
# 5) PSF 测光
#    使用 MoffatPSF 模型，初始通量来自孔径测光
#====================================================

def generate_moffat_psf(x0, y0, alpha, beta, flux, size=20):
    x = np.linspace(x0 - size//2, x0 + size//2, size)
    y = np.linspace(y0 - size//2, y0 + size//2, size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    psf = flux * (1 + (r / alpha)**2)**(-beta)
    return X, Y, psf


if sources is not None and len(sources) > 0:
    # 子样本选择（当前使用所有源，可根据需要调整）
    sub_sample = sources

    print(f"用于 PSF 拟合的子样本源数量：{len(sub_sample)}")

    if len(sub_sample) == 0:
        print("没有合适的源用于 PSF 参数测量。")
    else:
        # (a) 初始化 MoffatPSF 模型，允许 alpha 和 beta 拟合
        fitter = LevMarLSQFitter()
        psf_model = MoffatPSF(alpha=1.0, beta=2.5)  # Moffat 参数初始猜测
        psf_model.x_0.fixed = False
        psf_model.y_0.fixed = False
        psf_model.flux.fixed = False
        psf_model.alpha.fixed = False
        psf_model.beta.fixed = False

        psfphot = PSFPhotometry(
            psf_model=psf_model,
            fit_shape=(13, 13),
            finder=daofind,  # 重用 DAOStarFinder
            aperture_radius=3,
            progress_bar=False
        )

        # 设置初始参数，使用孔径测光的结果作为初始通量
        init_params = QTable()
        init_params['x'] = sub_sample['xcentroid'].values
        init_params['y'] = sub_sample['ycentroid'].values
        init_params['flux'] = flux_aperture  # 使用孔径测光的通量

        # (b) 对子样本进行 PSF 拟合，测量 alpha 和 beta
        fwhm_measure = psfphot(data_sub_masked, init_params=init_params).to_pandas()
        initial_flux_sum = fwhm_measure['flux_fit'].sum()
        print(f"初步 PSF 拟合总通量：{initial_flux_sum}")
        print(fwhm_measure.columns)

        # (c) 使用测量的 alpha 和 beta 更新 PSF 模型，并对所有源进行 PSF 测光
        if 'alpha_fit' in fwhm_measure.columns and 'beta_fit' in fwhm_measure.columns:
            alpha = np.median(fwhm_measure['alpha_fit'])
            beta = np.median(fwhm_measure['beta_fit'])
            print(f"测量的 PSF alpha: {alpha:.2f}, beta: {beta:.2f}")

            # 更新 PSF 模型
            psf_model = MoffatPSF(alpha=alpha, beta=beta)
            psf_model.x_0.fixed = False
            psf_model.y_0.fixed = False
            psf_model.flux.fixed = False
            psf_model.alpha.fixed = False  # 固定 alpha 和 beta
            psf_model.beta.fixed = False
            daogroup = SourceGrouper(5)  # 分组附近的源

            # 使用更新参数重新检测源
            daofind = DAOStarFinder(threshold=threshold, fwhm=2.0)  # 可根据需要调整
            sources_updated = daofind(data_sub_masked)
            num_sources_updated = len(sources_updated)
            print(f"使用更新参数检测到的源数量：{num_sources_updated}")

            photometry = PSFPhotometry(
                grouper=daogroup,
                psf_model=psf_model,
                fit_shape=(13, 13),
                finder=daofind,
                aperture_radius=3,
                progress_bar=True,
                xy_bounds=10
            )

            # 设置所有源的初始参数
            init_params_all = QTable()
            init_params_all['x'] = sources_updated['xcentroid']
            init_params_all['y'] = sources_updated['ycentroid']
            init_params_all['flux'] = flux_aperture  # 使用孔径测光通量作为初始值

            # 执行 PSF 测光
            psf_phot_results = photometry(data_sub_masked, init_params=init_params_all).to_pandas()
            updated_flux_sum = psf_phot_results['flux_fit'].sum()
            print(f"更新后的 PSF 拟合总通量：{updated_flux_sum}")
            print("PSF 测光结果（前几行）：")
            print(psf_phot_results.head())

            # 绘制图像和等值线
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(data_sub, origin='lower', cmap='gray')

            for idx, row in psf_phot_results.iterrows():
                x0, y0 = row['x_fit'], row['y_fit']
                alpha, beta = row['alpha_fit'], row['beta_fit']
                flux = row['flux_fit']
                X, Y, psf = generate_moffat_psf(x0, y0, alpha, beta, flux)
                levels = [0.1 * psf.max(), 0.5 * psf.max()]  # 10% 和 50% 亮度水平
                ax.contour(X, Y, psf, levels=levels, colors='lime', linewidths=1)

            plt.tight_layout()
            plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/psf_photometry_moffat.png')
        else:
            print("未找到 'gamma_fit' 或 'alpha_fit' 列。请检查 psf_model 参数是否被拟合。")
else:
    print("源表中缺少 'roundness1'、'roundness2' 或 'flux' 列，跳过 PSF 测光。")
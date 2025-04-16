import numpy as np
from imageio import imread
from astropy.stats import sigma_clipped_stats
from photutils.aperture import aperture_photometry, CircularAperture
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.psf import PSFPhotometry, CircularGaussianPRF, SourceGrouper
from astropy.io import fits
from astropy.table import QTable
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import pdb
#====================================================
# 1) 读取FITS图像并创建掩码
#====================================================

filename = "/home/bingxing2/ailab/group/ai4astro/Datasets/Astro_SR/origin_fits/hst_15851_13_acs_wfc_f814w_je5613_drc.fits"
hdul = fits.open(filename)
hdul.info()

data = hdul[1].data.astype(float)  # Ensure data is float for NaN handling
mask = data != 0  # True for valid regions (non-zero), False for invalid (zero)
hdul.close()

#====================================================
# 2) 背景扣除
#====================================================

box_size = 64
bkg_estimator = MedianBackground()
bkg = Background2D(data, box_size, filter_size=(3, 3), bkg_estimator=bkg_estimator, mask=~mask)  # ~mask: True for invalid regions
data_sub = data - bkg.background

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# norm = ImageNormalize(data, interval=ZScaleInterval())
# ax1.imshow(data, origin='lower', cmap='gray', norm=norm)
# ax1.set_title('HST F814W Data (before background subtraction)')

norm_sub = ImageNormalize(data_sub, interval=ZScaleInterval())
# ax2.imshow(data_sub, origin='lower', cmap='gray', norm=norm_sub)
# ax2.set_title('HST F814W Data (background-subtracted)')

# plt.tight_layout()
# plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/background_subtraction.png')

#====================================================
# 3) 源检测（DAOStarFinder）
#====================================================

# Mask invalid regions with NaN to avoid detecting false sources
data_sub_masked = np.where(mask, data_sub, np.nan)

mean_val, median_val, std_val = sigma_clipped_stats(data_sub_masked, sigma=3.0)
threshold = 8.0 * std_val  # Detection threshold: 8σ, adjustable
fwhm_guess = 2.0           # Initial FWHM guess for HST/ACS, ~3 pixels

daofind = DAOStarFinder(threshold=threshold, fwhm=fwhm_guess)
sources_tbl = daofind(data_sub_masked)

if sources_tbl is None or len(sources_tbl) == 0:
    print("No sources detected. Adjust 'threshold' or 'fwhm_guess'.")
    sources = None
else:
    sources = sources_tbl.to_pandas()
    print(f"Detected {len(sources)} sources.")
    print(sources.head())

#====================================================
# 4) 孔径测光 (Aperture Photometry)
#====================================================

if sources is not None and len(sources) > 0:
    positions = np.transpose((sources['xcentroid'].values, sources['ycentroid'].values))
    aperture_radius = 2.0  # Aperture radius in pixels

    apertures = CircularAperture(positions, r=aperture_radius)
    phot_table = aperture_photometry(data_sub, apertures, mask=~mask)  # ~mask: True for invalid regions

    # Extract photometry results
    flux_aperture = phot_table['aperture_sum']

    print("Aperture Photometry Results:")
    print("Number of detected sources:", len(flux_aperture))
    print("Total Flux:", flux_aperture.sum())

    # Visualize detected sources with red circles
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(data_sub, origin='lower', cmap='gray', norm=norm_sub)
    ax.set_title('Aperture Photometry (red circles)')
    for x, y in positions:
        circ = Circle((x, y), radius=aperture_radius, edgecolor='red', facecolor='none', lw=1)
        ax.add_patch(circ)
    plt.tight_layout()
    plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/aperture_photometry.png')
#====================================================
# 5) PSF测光
#    先根据源表的roundness, flux等筛选子样本，测量FWHM
#    然后对所有源进行PSFPhotometry
#====================================================

if sources is not None and len(sources) > 0:
    # Check if required columns exist for source filtering
    if {'roundness1', 'roundness2', 'flux'}.issubset(sources.columns):
        # Sub-sample selection (currently using all sources; adjust as needed)
        sub_sample = sources
        #sub_sample = sources[(sources['roundness1'].abs() < 0.5) & (sources['roundness2'].abs() < 0.5)]  # Optionally filter: e.g., based on roundness/flux

        print(f"Sub-sample sources for PSF FWHM fit: {len(sub_sample)}")

        if len(sub_sample) == 0:
            print("No suitable sources for PSF FWHM measurement.")
        else:
            # (a) Initialize PSF model with fittable FWHM
            fitter = LevMarLSQFitter()
            psf_model = CircularGaussianPRF(fwhm=3.0)  # Initial FWHM guess
            psf_model.x_0.fixed = False
            psf_model.y_0.fixed = False
            psf_model.flux.fixed = False
            psf_model.fwhm.fixed = False

            psfphot = PSFPhotometry(
                psf_model=psf_model,
                fit_shape=(13, 13),
                finder=daofind,  # Reuse DAOStarFinder
                aperture_radius=3,
                progress_bar=False
            )

            init_params = QTable()
            init_params['x'] = sub_sample['xcentroid'].values
            init_params['y'] = sub_sample['ycentroid'].values

            # (b) Fit PSF to sub-sample to measure FWHM
            fwhm_measure = psfphot(data_sub_masked, init_params=init_params).to_pandas()
            initial_flux_sum = fwhm_measure['flux_fit'].sum()
            print(f"Initial PSF fit total flux: {initial_flux_sum}")
            print(fwhm_measure.columns)

            if 'fwhm_fit' in fwhm_measure.columns:
                # (c) Sigma-clip FWHM measurements to get a robust mean
                fwhm, _, fwhm_std = sigma_clipped_stats(fwhm_measure['fwhm_fit'], sigma=3, maxiters=10)
                print(f"Measured PSF FWHM: {fwhm:.2f} ± {fwhm_std:.2f}")

                # (d) Update PSF model with measured FWHM and perform PSF photometry on all sources
                psf_model = CircularGaussianPRF(fwhm=fwhm)
                psf_model.x_0.fixed = False
                psf_model.y_0.fixed = False
                psf_model.flux.fixed = False
                psf_model.fwhm.fixed = False
                daogroup = SourceGrouper(5)  # Group nearby sources

                # Redetect sources with updated FWHM
                daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold)
                sources_updated = daofind(data_sub_masked)
                num_sources_updated = len(sources_updated)
                print(f"Number of sources detected with updated FWHM: {num_sources_updated}")

                photometry = PSFPhotometry(
                    grouper=daogroup,
                    psf_model=psf_model,
                    fit_shape=(13, 13),
                    finder=daofind,
                    aperture_radius=3,
                    progress_bar=True,
                    xy_bounds=10
                )

                init_params_all = QTable()
                init_params_all['x'] = sources_updated['xcentroid']
                init_params_all['y'] = sources_updated['ycentroid']
                init_params_all['flux'] = sources_updated['flux']  # Initial flux from DAOStarFinder

                # Perform PSF photometry
                psf_phot_results = photometry(data_sub_masked, init_params=init_params_all).to_pandas()
                updated_flux_sum = psf_phot_results['flux_fit'].sum()
                print(f"Updated PSF fit total flux: {updated_flux_sum}")
                print("PSF Photometry Results (head):")
                print(psf_phot_results.head())

                # Visualize PSF photometry results
                fig, ax = plt.subplots(figsize=(15, 15))
                ax.imshow(data_sub, origin='lower', cmap='gray', norm=norm_sub)
                ax.set_title('PSF Photometry - each star with its own FWHM')

                for idx, row in psf_phot_results.iterrows():
                    x = row['x_fit']
                    y = row['y_fit']
                    local_fwhm = row['fwhm_fit']  # Use fitted FWHM per source
                    circ = Circle((x, y), radius=local_fwhm, edgecolor='lime', facecolor='none', lw=1)
                    ax.add_patch(circ)

                plt.tight_layout()
                plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/psf_photometry_localFwhm.png')
            else:
                print("No 'fwhm_fit' column found. Check if psf_model.fwhm.fixed is True.")
    else:
        print("No 'roundness1', 'roundness2', or 'flux' columns in source table. Skipping PSF photometry.")
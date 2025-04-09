import os
from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from regions import PolygonSkyRegion
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from shapely.geometry import Polygon
import shapely.wkt
import pdb
TEMP_NUM = 200
data_dir = "/home/bingxing2/ailab/group/ai4astro/Datasets/Astro_SR/psf"
output_file = "/home/bingxing2/ailab/scxlab0061/Astro_SR/data_process/split_file/datasetlist.txt"
fits_files = []
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith("_drc.fits.gz"):
            fits_files.append(os.path.join(root, file))
            if len(fits_files) >= TEMP_NUM:
                break
    if len(fits_files) >= TEMP_NUM:
        break
with open(output_file, 'w') as f:
    for fits_filepath in tqdm(fits_files, desc="Processing FITS files"):
        try:

            hdu = fits.open(fits_filepath)[1]
            wcs = WCS(hdu.header)
            x = [1, hdu.data.shape[1], hdu.data.shape[1], 1]
            y = [1, 1, hdu.data.shape[0], hdu.data.shape[0]]
            ra, dec = wcs.all_pix2world(x, y, 1)
            corners = list(zip(ra, dec))
            vertices = SkyCoord([c[0] * u.deg for c in corners],
                                [c[1] * u.deg for c in corners], frame='icrs')
            poly_region = PolygonSkyRegion(vertices=vertices)
            ra = poly_region.vertices.ra.deg
            dec = poly_region.vertices.dec.deg
            region = Polygon(np.column_stack((ra, dec)))
            f.write(f"{fits_filepath}:{region.wkt}\n")

        except Exception as e:
            print(f"Error processing {fits_filepath}: {e}")
            continue
print(f"Done! Results saved to {output_file}")
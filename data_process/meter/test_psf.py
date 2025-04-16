import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.psf import IntegratedGaussianPRF, PSFPhotometry
from photutils.background import Background2D, MedianBackground
import pytest
from unittest.mock import patch, MagicMock
from Astro_SR.data_process.meter.psf_patchify import perform_psf_photometry

# Import the function to test


class TestPSFPhotometry(unittest.TestCase):
    
    def setUp(self):
        # Create synthetic test image with stars
        self.image_size = 100
        self.image = np.zeros((self.image_size, self.image_size))
        self.mask = np.ones((self.image_size, self.image_size), dtype=bool)
        
        # Add background with some noise
        self.background_level = 10
        self.image += self.background_level + np.random.normal(0, 0.5, self.image.shape)
        
        # Add some stars with known positions and fluxes
        self.star_positions = [
            (20, 20, 100),  # x, y, flux
            (60, 30, 150),
            (40, 70, 200),
            (80, 60, 180)
        ]
        
        # Create Gaussian PSF stars
        for x, y, flux in self.star_positions:
            y_grid, x_grid = np.mgrid[0:self.image_size, 0:self.image_size]
            sigma = 2.0
            star = flux * np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            self.image += star
    
    def test_star_detection_and_photometry(self):
        """Test if stars are correctly detected and measured with perform_psf_photometry"""
        # We need to patch pdb.set_trace to avoid debugging during tests
        with patch('pdb.set_trace'):
            stars = perform_psf_photometry(self.image, self.mask)
        
        # Check if the right number of stars are detected
        self.assertEqual(len(stars), len(self.star_positions), 
                        f"Expected {len(self.star_positions)} stars, got {len(stars)}")
        
        # Create dictionaries for easier comparison
        expected_stars = {(x, y): flux for x, y, flux in self.star_positions}
        found_stars = {(round(s['x']), round(s['y'])): s['flux'] for s in stars}
        
        # Check if all expected stars are found (with some tolerance for position)
        for pos, flux in expected_stars.items():
            # Find the closest detected star
            closest_dist = float('inf')
            closest_star = None
            
            for star_pos, star_flux in found_stars.items():
                dist = np.sqrt((pos[0] - star_pos[0])**2 + (pos[1] - star_pos[1])**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_star = (star_pos, star_flux)
            
            # Assert that a star was found nearby (within 5 pixels)
            self.assertLessEqual(closest_dist, 5, 
                               f"No star found near expected position {pos}")
            
            # Check that flux is reasonably close (within 30%)
            # PSF fitting can give different values than input due to noise and fitting process
            relative_flux_diff = abs(closest_star[1] - flux) / flux
            self.assertLessEqual(relative_flux_diff, 0.3, 
                               f"Flux for star at {pos} is too different: expected {flux}, got {closest_star[1]}")
    
    def test_with_masked_regions(self):
        """Test how the function handles masked regions"""
        # Create a mask that excludes one star
        masked_image = self.mask.copy()
        masked_image[10:30, 10:30] = False  # Mask the first star
        
        # Run photometry with the masked image
        with patch('pdb.set_trace'):
            stars = perform_psf_photometry(self.image, masked_image)
        
        # We should have one less star
        self.assertEqual(len(stars), len(self.star_positions) - 1,
                       f"Expected {len(self.star_positions) - 1} stars when one is masked, got {len(stars)}")
        
        # The masked star should not be in the results
        for star in stars:
            # Check it's not near the masked star position (x=20, y=20)
            dist = np.sqrt((star['x'] - 20)**2 + (star['y'] - 20)**2)
            self.assertGreater(dist, 5, "Masked star was detected")
    
    def test_no_stars_detected(self):
        """Test behavior when no stars are detected"""
        # Create an image with just noise, no stars
        noise_image = np.random.normal(self.background_level, 0.5, (self.image_size, self.image_size))
        
        # Mock DAOStarFinder to return None
        with patch('photutils.detection.DAOStarFinder.find_stars', return_value=None):
            with patch('pdb.set_trace'):
                stars = perform_psf_photometry(noise_image, self.mask)
        
        # Check that an empty list is returned
        self.assertEqual(len(stars), 0, "Expected empty list when no stars detected")
    
    def test_daofind_vs_psf_measurements(self):
        """Compare results from DAOStarFinder with PSF photometry results"""
        # Use a modified version of perform_psf_photometry that captures intermediate results
        def modified_perform_psf_photometry(image, mask):
            bkg = Background2D(image, (64, 64), filter_size=(3, 3), 
                              bkg_estimator=MedianBackground(), mask=~mask)
            data_sub = image - bkg.background
            data_sub_masked = np.where(mask, data_sub, np.nan)
            
            mean, median, std = sigma_clipped_stats(data_sub_masked, sigma=3.0)
            daofind = DAOStarFinder(fwhm=3.0, threshold=8.0 * std)
            sources_tbl = daofind(data_sub_masked)
            
            if sources_tbl is None or len(sources_tbl) == 0:
                return [], None, None
            
            # Rename columns for PSF photometry
            sources_tbl.rename_column('xcentroid', 'x_0')
            sources_tbl.rename_column('ycentroid', 'y_0')
            sources_tbl.rename_column('flux', 'flux_0')
            
            # Execute PSF photometry
            psf_model = IntegratedGaussianPRF(sigma=1.0)
            psf_model.sigma.fixed = False
            photometry = PSFPhotometry(psf_model, fit_shape=(11, 11), aperture_radius=5.0)
            phot_table = photometry(data_sub_masked, init_params=sources_tbl[['x_0', 'y_0', 'flux_0']])
            
            # Extract star information
            stars = []
            for row in phot_table:
                x = float(row['x_fit'])
                y = float(row['y_fit'])
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[int(np.round(y)), int(np.round(x))]:
                        star = {
                            'x': x,
                            'y': y,
                            'flux': float(row['flux_fit'])
                        }
                        stars.append(star)
            
            return stars, sources_tbl, phot_table
        
        # Get both DAOStarFinder and PSF photometry results
        with patch('pdb.set_trace'):
            stars, dao_sources, psf_results = modified_perform_psf_photometry(self.image, self.mask)
        
        # Verify we have results
        self.assertIsNotNone(dao_sources, "DAOStarFinder didn't find any stars")
        self.assertIsNotNone(psf_results, "PSF photometry didn't produce results")
        
        # Compare number of stars
        self.assertEqual(len(stars), len(self.star_positions),
                       f"Expected {len(self.star_positions)} stars, got {len(stars)}")
        
        # Compare initial DAOStarFinder positions with final PSF positions
        for i, star in enumerate(stars):
            # Find corresponding DAOStarFinder source
            dao_idx = np.argmin([(star['x'] - src['x_0'])**2 + (star['y'] - src['y_0'])**2 
                              for src in dao_sources])
            
            dao_x = dao_sources[dao_idx]['x_0']
            dao_y = dao_sources[dao_idx]['y_0']
            
            # Check that PSF fitting refined the position (should be close but not identical)
            self.assertNotEqual((dao_x, dao_y), (star['x'], star['y']),
                              "PSF fitting did not refine the position")
            
            # But they should be close
            dist = np.sqrt((dao_x - star['x'])**2 + (dao_y - star['y'])**2)
            self.assertLessEqual(dist, 3, f"PSF position too different from DAOStarFinder position")
            
            # Check that the fitted flux is different from the initial flux
            dao_flux = dao_sources[dao_idx]['flux_0']
            self.assertNotEqual(dao_flux, star['flux'],
                              "PSF fitting did not refine the flux")


if __name__ == '__main__':
    unittest.main()
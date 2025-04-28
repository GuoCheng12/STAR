import numpy as np
import sep
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# Lock for thread-safe operations
lock = threading.Lock()

def preprocess_gt_data(gt_path, save_dir, failed_paths, fw_initial=1, fh_initial=1, fw_retry=3, fh_retry=3):
    """
    Preprocess GT images, detect sources, and compute flux, saving results to .npy files.
    If photometry fails, retry with adjusted fw and fh parameters.
    Record failed paths in the shared list. Store successful fw and fh parameters.

    Parameters:
    - gt_path: Path to the GT image .npy file.
    - save_dir: Directory to save preprocessed results.
    - failed_paths: List to record paths of files that failed photometry.
    - fw_initial: Initial fw parameter for background subtraction.
    - fh_initial: Initial fh parameter for background subtraction.
    - fw_retry: fw parameter to retry if initial attempt fails.
    - fh_retry: fh parameter to retry if initial attempt fails.
    """
    try:
        # Load GT image data
        gt_data = np.load(gt_path, allow_pickle=True).item()
        gt = gt_data['image']  # GT image
        mask = gt_data.get('mask', None)  # Mask, if exists

        # Handle mask (create a full True mask if none exists)
        if mask is None:
            mask = np.ones_like(gt, dtype=bool)
        else:
            mask = ~mask  # SEP uses True for masked regions

        # Attempt photometry with initial parameters
        sources, flux, fw_used, fh_used = attempt_photometry(gt, mask, fw_initial, fh_initial)
        success = True if sources is not None and flux is not None else False

        if not success:
            # Retry with adjusted fw and fh if initial attempt fails
            print(f"Retrying with fw={fw_retry}, fh={fh_retry} for {gt_path}")
            sources, flux, fw_used, fh_used = attempt_photometry(gt, mask, fw_retry, fh_retry)
            success = True if sources is not None and flux is not None else False

        if not success:
            # If still failing, set flux to 0 and record the path
            print(f"Failed to measure flux for {gt_path}, setting flux to 0")
            sources = np.array([])  # Empty array
            flux = np.array([0.0])
            fw_used, fh_used = None, None  # No successful parameters
            with lock:
                failed_paths.append(gt_path)
        else:
            # Store successful fw and fh parameters
            gt_data['fw_used'] = fw_used
            gt_data['fh_used'] = fh_used

        # Save sources and flux to data dictionary
        gt_data['sources'] = sources
        gt_data['flux'] = flux

        # Save to new .npy file
        save_path = os.path.join(save_dir, os.path.basename(gt_path))
        np.save(save_path, gt_data)
        print(f"Preprocessing completed, saved to: {save_path}")

    except Exception as e:
        print(f"Error processing {gt_path}: {e}")
        with lock:
            failed_paths.append(gt_path)

def attempt_photometry(image, mask, fw, fh):
    """
    Attempt photometry on the image, returning sources, flux, and the fw/fh used.
    Returns None if photometry fails.

    Parameters:
    - image: Input image data.
    - mask: Mask for the image.
    - fw: Filter width for background subtraction.
    - fh: Filter height for background subtraction.
    """
    try:
        # Background subtraction
        bkg = sep.Background(image, mask=mask, bw=64, bh=64, fw=fw, fh=fh)
        image_sub = image - bkg.back()

        # Source detection
        sources = sep.extract(image_sub, 1.5, err=bkg.rms(), mask=mask)

        # Photometry: compute flux
        flux, fluxerr, flag = sep.sum_ellipse(
            image_sub, sources['x'], sources['y'],
            sources['a'], sources['b'], sources['theta'],
            2.5, err=bkg.globalrms, mask=mask
        )

        # Filter NaN values
        valid_idx = ~np.isnan(flux)
        sources_cleaned = sources[valid_idx]
        flux_cleaned = flux[valid_idx]

        return sources_cleaned, flux_cleaned, fw, fh
    except Exception as e:
        print(f"Photometry failed: {e}")
        return None, None, None, None
    
if __name__ == "__main__":
    # Define input and output paths
    path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy/train_hr_patch"
    save_dir = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy/train_hr_patch"  # Update this to your desired save directory

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get all .npy files in the directory
    npy_files = glob.glob(os.path.join(path, "*.npy"))
    failed_paths = []
    # Use ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(preprocess_gt_data, gt_path, save_dir, failed_paths) for gt_path in npy_files]
        progress = tqdm(total=len(futures), desc="Processing .npy files")

        for future in as_completed(futures):
            try:
                future.result()  # Wait for each thread to complete
                progress.update(1)
            except Exception as e:
                print(f"Error in thread: {e}")
    progress.close()
    # Save failed paths to a .txt file
    failed_txt_path = os.path.join(save_dir, "failed_samples.txt")
    with open(failed_txt_path, "w") as f:
        for path in failed_paths:
            f.write(f"{path}\n")
    print(f"Failed samples saved to: {failed_txt_path}")
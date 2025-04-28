import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import pdb
# Specify the .npy file path
npy_path = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataset_gaussian_airy/train_hr_patch/hst_10499_02_acs_wfc_f814w_j9bn02_drc_padded_hr_hr_patch_909.npy"

# Load the .npy file
data = np.load(npy_path, allow_pickle=True).item()
image = data['image'].astype(np.float32)  # Extract the original image
attn_map = data['attn_map']  # Extract the attention map
# Apply Z-scale normalization (for the original image)
zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(image)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), dpi=300)

# Subplot 1: Original Image
ax1.imshow(image, interpolation='nearest', cmap='gray',
           vmin=vmin, vmax=vmax, origin='lower')
ax1.set_title('Original Image')

# Subplot 2: Attention Map (Log Scale)
ax2.imshow(attn_map, cmap='hot', origin='lower')
ax2.set_title('Attention Map (Log Scale)')

# Save the image
plt.tight_layout()
plt.savefig('/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/verify_attn_map.png')
plt.close()

print(f"Visualization saved to /home/bingxing2/ailab/scxlab0061/Astro_SR/vis/meter/verify_attn_map.png")
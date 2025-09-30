import torch
import torch.nn as nn
from scipy.ndimage import zoom
from astropy.io import fits
import numpy as np
import torch.nn.functional as F

def estimate_background(image, box_size, filter_size):
    H, W = image.shape
    boxes = image.unfold(0, box_size, box_size).unfold(1, box_size, box_size)
    median_values = boxes.contiguous().view(-1, box_size, box_size).median(dim=0)[0]
    n_boxes_h = (H + box_size - 1) // box_size
    n_boxes_w = (W + box_size - 1) // box_size
    median_values = median_values.view(n_boxes_h, n_boxes_w)
    bkg = median_values.repeat_interleave(box_size, dim=0).repeat_interleave(box_size, dim=1)
    bkg = bkg[:H, :W]
    bkg_smoothed = F.avg_pool2d(bkg.unsqueeze(0).unsqueeze(0), filter_size, stride=1, padding=filter_size[0] // 2)
    return bkg_smoothed.squeeze(0).squeeze(0)


def sigma_clipped_stats(data, sigma=3.0, max_iter=5):
    mask = torch.ones_like(data, dtype=torch.bool)
    for _ in range(max_iter):
        data_filtered = data[mask]
        mean_val = data_filtered.mean()
        std_val = data_filtered.std(unbiased=True)
        mask = (data - mean_val).abs() <= sigma * std_val
    data_filtered = data[mask]
    mean_val = data_filtered.mean()
    median_val = torch.median(data_filtered)
    std_val = data_filtered.std(unbiased=True)
    return mean_val, median_val, std_val


def find_peaks_vectorized(data, threshold, fwhm_guess):
    padded_data = F.pad(data.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    local_maxima = (data == F.max_pool2d(padded_data, kernel_size=3, stride=1)).squeeze()
    peaks = torch.where((data > threshold) & local_maxima)
    H, W = data.shape
    mask = torch.zeros_like(data, dtype=torch.bool, device=data.device, requires_grad=False)
    if len(peaks[0]) > 0:
        y_centers, x_centers = peaks
        radius = 4
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=data.device),
            torch.arange(-radius, radius + 1, device=data.device),
            indexing='ij'
        )
        y_offsets_expanded = offsets_y.flatten()[None, :]  # Shape: [1, (2*radius+1)**2]
        x_offsets_expanded = offsets_x.flatten()[None, :]  # Shape: [1, (2*radius+1)**2]
        y_indices = y_centers[:, None] + y_offsets_expanded  # Shape: [num_peaks, (2*radius+1)**2]
        x_indices = x_centers[:, None] + x_offsets_expanded  # Shape: [num_peaks, (2*radius+1)**2]
        y_indices = y_indices.flatten()
        x_indices = x_indices.flatten()
        valid = (y_indices >= 0) & (y_indices < H) & (x_indices >= 0) & (x_indices < W)
        y_indices, x_indices = y_indices[valid], x_indices[valid]
        mask[y_indices, x_indices] = True
    return mask.long()


class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, gt):
        l1 = self.l1(pred, gt)
        pred = pred.squeeze()
        gt = gt.squeeze()

        # subtract bkg
        box_size = 16
        filter_size = (3, 3)
        bkg = estimate_background(gt, box_size, filter_size)
        data = gt - bkg
        pred = pred - bkg

        # find star
        mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=3.0)
        threshold = 11.0 * std_val
        fwhm_guess = 1.0
        mask = find_peaks_vectorized(data, threshold, fwhm_guess)

        # photometry
        phot1 = torch.sum(data * mask)
        phot2 = torch.sum(pred * mask)
        # l2 = self.l1(pred * mask, data * mask)

        # photometry
        loss = torch.sum(torch.abs(phot1 - phot2))
        loss = torch.sum(torch.abs(phot1 - phot2)) / torch.sum(torch.abs(phot1))

        return loss
        # return 0.5 * l2 + 0.5 * l1


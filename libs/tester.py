import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde, linregress
from tqdm import tqdm
import torch.distributed as dist
from .utils import vis_astro_SR, evaluate_metric_SR
import random
import sep
from tqdm import tqdm
from matplotlib.patches import Ellipse
class Tester(object):
    def __init__(self, 
                 model, 
                 evalloader, 
                 local_rank=0,
                 ddp=False,
                 visualize=False,
                 vis_dir=None,
                 logger=False):
        self.logger = logger
        self.model = model
        self.evalloader = evalloader
        self.visualize = visualize
        self.vis_dir = vis_dir
        self.ddp = ddp
        self.local_rank = local_rank

        if ddp and type(self.model) is not nn.parallel.DistributedDataParallel:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[local_rank % torch.cuda.device_count()],
                                                             find_unused_parameters=False)  
        if self.vis_dir is not None and self.local_rank == 0:
            os.makedirs(vis_dir, exist_ok=True)
            
    def test(self):
        self.model.eval()
        total_ssim = 0.0
        total_psnr = 0.0
        num_samples = 0
        flux_losses = []  # List to store flux differences for each sample
        problematic_samples = []  # List to store problematic samples

        for datalist in tqdm(self.evalloader):  
            infer_datalist = datalist.copy()
            for key in infer_datalist.keys():
                if type(infer_datalist[key]) is torch.Tensor:
                    infer_datalist[key] = infer_datalist[key].to('cuda')
            with torch.no_grad():
                results = self.model(infer_datalist['input'], infer_datalist)
                results = {key: results[key].cpu() if type(results[key]) is torch.Tensor else results[key] for key in results.keys()}
            batch_ssim, batch_psnr = evaluate_metric_SR(results['pred_img'], datalist['hr'], datalist['mask'])
            total_ssim += batch_ssim * len(datalist['hr'])
            total_psnr += batch_psnr * len(datalist['hr'])
            num_samples += len(datalist['hr'])

            # Photometry and Flux Consistency Loss Calculation
            for i in range(len(datalist['hr'])):
                try:
                    pred_img = results['pred_img'][i].numpy()
                    gt = datalist['hr'][i].numpy()
                    mask = datalist['mask'][i].numpy().astype(bool)
                    flux_gt, sources_gt = self.measure_flux(gt, mask)
                    flux_pred = self.measure_flux_with_gt_sources(pred_img, sources_gt, mask)
                    if len(flux_pred) == len(flux_gt):
                        flux_diff = np.abs(flux_pred - flux_gt)
                        loss = np.mean(flux_diff)
                        flux_losses.append(loss)
                    else:
                        print(f"Sample {i}: Flux lengths do not match, skipping.")
                        problematic_samples.append(i)
                except Exception as e:
                    print(f"Sample {i} encountered an error: {e}, skipping.")
                    problematic_samples.append(i)

        # Calculate average flux difference
        avg_flux_loss = np.mean(flux_losses) if flux_losses else 0.0

        # Output results
        avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0
        avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
        print(f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.4f}")
        print(f"Average Flux Consistency Loss: {avg_flux_loss:.4f}")
        print(f"Problematic samples: {problematic_samples}")

    def measure_flux(self, image, mask):
        """对图像进行源检测和测光"""
        image = image.squeeze()
        mask = mask.squeeze()
        try:
            bkg = sep.Background(image, mask=~mask)
            image_sub = image - bkg.back()
            sources = sep.extract(image_sub, 1.5, err=bkg.rms(), mask=~mask)
            flux, fluxerr, flag = sep.sum_ellipse(
                image_sub, sources['x'], sources['y'],
                sources['a'], sources['b'], sources['theta'],
                2.5, err=bkg.globalrms
            )
            valid_idx = ~np.isnan(flux)
            flux_cleaned = flux[valid_idx]
            sources_cleaned = sources[valid_idx]
            return flux_cleaned, sources_cleaned
        except Exception as e:
            raise e

    def measure_flux_with_gt_sources(self, image, sources, mask):
        """使用 gt 图像的源信息对图像进行测光"""
        image = image.squeeze()
        mask = mask.squeeze()
        try:
            bkg = sep.Background(image, mask=~mask)
            image_sub = image - bkg.back()
            flux, fluxerr, flag = sep.sum_ellipse(
                image_sub, sources['x'], sources['y'],
                sources['a'], sources['b'], sources['theta'],
                2.5, err=bkg.rms()
            )
            valid_idx = ~np.isnan(flux)
            flux_cleaned = flux[valid_idx]
            return flux_cleaned
        except Exception as e:
            raise e

    def eval(self):
        self.model.eval()
        total_ssim = 0.0
        total_psnr = 0.0
        num_samples = 0
        for datalist in self.evalloader:  
            infer_datalist = datalist.copy()
            for key in infer_datalist.keys():
                if type(infer_datalist[key]) is torch.Tensor:
                    infer_datalist[key] = infer_datalist[key].to('cuda')
            with torch.no_grad():
                results = self.model(infer_datalist['input'], infer_datalist)
                results = {key:results[key].cpu() if type(results[key]) is torch.Tensor else results[key] for key in results.keys()}
            batch_ssim, batch_psnr = evaluate_metric_SR(results['pred_img'], datalist['hr'], datalist['mask'])
            total_ssim += batch_ssim * len(datalist['hr'])
            total_psnr += batch_psnr * len(datalist['hr'])
            num_samples += len(datalist['hr'])
        if self.ddp:
            total_ssim_tensor = torch.tensor(total_ssim).to('cuda')
            total_psnr_tensor = torch.tensor(total_psnr).to('cuda')
            num_samples_tensor = torch.tensor(num_samples).to('cuda')
            dist.all_reduce(total_ssim_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_psnr_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)
            total_ssim = total_ssim_tensor.item()
            total_psnr = total_psnr_tensor.item()
            num_samples = num_samples_tensor.item()
        if self.local_rank == 0:
            avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0
            avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
            print(f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.4f}")
            if self.logger:
                result_epoch = f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.4f}"
                self.logger.info(result_epoch)
        if self.visualize and self.local_rank == 0:
                print("可视化")
                num_samples = len(datalist['hr'])
                indices = random.sample(range(num_samples), min(10, num_samples))
                for idx in indices:
                    pred = results['pred_img'][idx].numpy()  
                    target = datalist['hr'][idx].numpy()     
                    input_img = datalist['input'][idx].numpy()  
                    mask = datalist['mask'][idx].numpy()      # 传入掩码
                    name = datalist['filename'][idx]           
                    vis_astro_SR(pred, target, input_img, mask, name, self.vis_dir)  # 更新调用


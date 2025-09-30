import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde, linregress
from tqdm import tqdm
import torch.distributed as dist
from .utils import vis_astro_SR, evaluate_metric_SR,vis_astro_SR1
import random
import sep
import time
from astropy.modeling import models, fitting
import numpy as np

import sep
from tqdm import tqdm
import numpy as np
import torch
import random
from sep import extract  
from scipy.optimize import linear_sum_assignment  #

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

    def measure_shear(self, sub_image):
        model = models.Gaussian2D(amplitude=np.max(sub_image), x_mean=sub_image.shape[1]/2, y_mean=sub_image.shape[0]/2, x_stddev=5, y_stddev=5, theta=0)
        fitter = fitting.LevMarLSQFitter()
        y, x = np.mgrid[:sub_image.shape[0], :sub_image.shape[1]]
        fitted_model = fitter(model, x, y, sub_image)
        
        e1 = (fitted_model.x_stddev**2 - fitted_model.y_stddev**2) / (fitted_model.x_stddev**2 + fitted_model.y_stddev**2) * np.cos(2 * fitted_model.theta)
        e2 = (fitted_model.x_stddev**2 - fitted_model.y_stddev**2) / (fitted_model.x_stddev**2 + fitted_model.y_stddev**2) * np.sin(2 * fitted_model.theta)
        
        return e1, e2

    def eval_shear(self):
        self.model.eval()
        total_ssim = 0.0
        total_psnr = 0.0
        total_mae_e1 = 0.0
        total_mae_e2 = 0.0
        num_samples = 0
        for datalist in tqdm(self.evalloader):  
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
            
            batch_mae_e1 = 0.0
            batch_mae_e2 = 0.0
            batch_object_count = 0
            
            for idx in tqdm(range(len(datalist['hr']))):
                hr_img = datalist['hr'][idx].numpy().squeeze()
                pred_img = results['pred_img'][idx].numpy().squeeze()
                mask = datalist['mask'][idx].numpy().squeeze() 


                mask = mask.astype(bool)
                hr_bkg = sep.Background(hr_img, mask=~mask, bw=32, bh=32, fw=1, fh=1)
                hr_img_bkgsub = hr_img - hr_bkg.back() 
                
                pred_bkg = sep.Background(pred_img, mask=~mask, bw=32, bh=32, fw=1, fh=1)
                pred_img_bkgsub = pred_img - pred_bkg.back()
                
                hr_sources = extract(hr_img_bkgsub, thresh=1.5, err=hr_bkg.rms(), mask=~mask)  
                pred_sources = extract(pred_img_bkgsub, thresh=1.5, err=pred_bkg.rms(), mask=~mask)
                
                if len(hr_sources) > 0 and len(pred_sources) > 0:
                    cost_matrix = np.zeros((len(hr_sources), len(pred_sources)))
                    for i, hr_src in enumerate(hr_sources):
                        for j, pred_src in enumerate(pred_sources):
                            dx = hr_src['x'] - pred_src['x']
                            dy = hr_src['y'] - pred_src['y']
                            cost_matrix[i, j] = np.sqrt(dx**2 + dy**2) 
                    
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    for r, c in zip(row_ind, col_ind):
                        if cost_matrix[r, c] < 5.0:  
                            hr_a, hr_b = hr_sources[r]['a'], hr_sources[r]['b']  # major/minor
                            size = max(2 * max(hr_a, hr_b), 50) 
                            x_hr, y_hr = hr_sources[r]['x'], hr_sources[r]['y']
                            hr_sub = hr_img_bkgsub[int(y_hr - size/2):int(y_hr + size/2), int(x_hr - size/2):int(x_hr + size/2)]
                            
                            x_pred, y_pred = pred_sources[c]['x'], pred_sources[c]['y']
                            pred_sub = pred_img_bkgsub[int(y_pred - size/2):int(y_pred + size/2), int(x_pred - size/2):int(x_pred + size/2)]
                            if hr_sub.shape[0] > 0 and hr_sub.shape[1] > 0 and pred_sub.shape[0] > 0 and pred_sub.shape[1] > 0:
                                hr_e1, hr_e2 = self.measure_shear(hr_sub)
                                pred_e1, pred_e2 = self.measure_shear(pred_sub)
                                
                                batch_mae_e1 += np.abs(hr_e1 - pred_e1)
                                batch_mae_e2 += np.abs(hr_e2 - pred_e2)
                                batch_object_count += 1
            
            if batch_object_count > 0:
                batch_mae_e1 /= batch_object_count
                batch_mae_e2 /= batch_object_count
            total_mae_e1 += batch_mae_e1 * len(datalist['hr'])
            total_mae_e2 += batch_mae_e2 * len(datalist['hr'])
            
            num_samples += len(datalist['hr'])
            if self.visualize:
                idx=1
                pred = results['pred_img'][idx].numpy()  
                target = datalist['hr'][idx].numpy()     
                input_img = datalist['input'][idx].numpy()  
                mask = datalist['mask'][idx].numpy()  
                name = datalist['filename'][idx]           
                vis_astro_SR(pred, target, input_img, mask,name, self.vis_dir)
                vis_astro_SR1(pred, target, input_img, mask,name, self.vis_dir)
        if self.ddp:
            total_ssim_tensor = torch.tensor(total_ssim).to('cuda')
            total_psnr_tensor = torch.tensor(total_psnr).to('cuda')
            total_mae_e1_tensor = torch.tensor(total_mae_e1).to('cuda')
            total_mae_e2_tensor = torch.tensor(total_mae_e2).to('cuda')
            num_samples_tensor = torch.tensor(num_samples).to('cuda')
            dist.all_reduce(total_ssim_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_psnr_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_mae_e1_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_mae_e2_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)
            total_ssim = total_ssim_tensor.item()
            total_psnr = total_psnr_tensor.item()
            total_mae_e1 = total_mae_e1_tensor.item()
            total_mae_e2 = total_mae_e2_tensor.item()
            num_samples = num_samples_tensor.item()
        if self.local_rank == 0:
            avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0
            avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
            avg_mae_e1 = total_mae_e1 / num_samples if num_samples > 0 else 0.0
            avg_mae_e2 = total_mae_e2 / num_samples if num_samples > 0 else 0.0
            print(f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.4f}")
            print(f"Average MAE e1: {avg_mae_e1:.4f}, Average MAE e2: {avg_mae_e2:.4f}")
            if self.logger:
                result_epoch = f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.4f}, Avg MAE e1: {avg_mae_e1:.4f}, Avg MAE e2: {avg_mae_e2:.4f}"
                self.logger.info(result_epoch)
        if self.visualize and self.local_rank == 0:
            print("Visualization")
            num_samples = len(datalist['hr'])
            indices = random.sample(range(num_samples), min(10, num_samples))
            for idx in indices:
                pred = results['pred_img'][idx].numpy()  
                target = datalist['hr'][idx].numpy()     
                input_img = datalist['input'][idx].numpy()  
                mask = datalist['mask'][idx].numpy()     
                name = datalist['filename'][idx]           
                vis_astro_SR(pred, target, input_img, mask, name, self.vis_dir) 
        return avg_ssim, avg_psnr, avg_mae_e1, avg_mae_e2






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
                a_time = time.time()
                results = self.model(infer_datalist['input'], infer_datalist)
                b_time = time.time()
                # print('final time',(b_time-a_time))
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
                    # mask = np.transpose(mask, (1, 2, 0))#asrtosr
                    flux_gt, sources_gt = self.measure_flux(gt, mask)
                    flux_pred1, sources_pred1 = self.measure_flux(pred_img, mask) 
                    # pred_img = np.transpose(pred_img, (1, 2, 0))#asrtosr
                    flux_pred = self.measure_flux_with_gt_sources(pred_img, sources_gt, mask)
                    if len(flux_pred) == len(flux_gt):
                        flux_diff = np.abs(flux_pred - flux_gt)
                        loss = np.mean(flux_diff)
                        flux_losses.append(loss)
                    else:
                        print(f"Sample {i}: Flux lengths do not match, skipping.")
                        problematic_samples.append(i)
                    if self.visualize:
                        pred = results['pred_img'][i].numpy()  
                        target = datalist['hr'][i].numpy()     
                        input_img = datalist['input'][i].numpy()  
                        mask = datalist['mask'][i].numpy()      
                        name = datalist['filename'][i]           
        
                        vis_astro_SR(pred, target, input_img, mask, name, self.vis_dir)  
      
                except Exception as e:
                    import pdb
                    # pdb.set_trace()
                    print("Visualization")
                    pred = results['pred_img'][i].numpy()  
                    target = datalist['hr'][i].numpy()     
                    input_img = datalist['input'][i].numpy()  
                    mask = datalist['mask'][i].numpy()      
                    name = datalist['filename'][i]           
                    vis_dir1 = 'error_data/'
                    vis_astro_SR(pred, target, input_img, mask, name, vis_dir1) 

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
        if self.visualize and self.local_rank == 0:
            print("Visualization")
            num_samples = len(datalist['hr'])
            indices = random.sample(range(num_samples), min(10, num_samples))
            for idx in indices:
                pred = results['pred_img'][idx].numpy()  
                target = datalist['hr'][idx].numpy()     
                input_img = datalist['input'][idx].numpy()  
                mask = datalist['mask'][idx].numpy()      
                name = datalist['filename'][idx]           
                vis_astro_SR(pred, target, input_img, mask, name, self.vis_dir) 

    def measure_flux(self, image, mask):
        """Perform source detection and photometry on the image"""
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
        """Use the source information of the gt image to perform image metering"""
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
            if self.visualize:
                idx=1
                pred = results['pred_img'][idx].numpy()  
                target = datalist['hr'][idx].numpy()     
                input_img = datalist['input'][idx].numpy()  
                mask = datalist['mask'][idx].numpy()  
                name = datalist['filename'][idx]           
                vis_astro_SR(pred, target, input_img, mask,name, self.vis_dir)
                vis_astro_SR1(pred, target, input_img, mask,name, self.vis_dir)
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
                print("Visualization")
                num_samples = len(datalist['hr'])
                indices = random.sample(range(num_samples), min(10, num_samples))
                for idx in indices:
                    pred = results['pred_img'][idx].numpy()  
                    target = datalist['hr'][idx].numpy()     
                    input_img = datalist['input'][idx].numpy()  
                    mask = datalist['mask'][idx].numpy()      
                    name = datalist['filename'][idx]           
                    vis_astro_SR(pred, target, input_img, mask, name, self.vis_dir)  
        return avg_ssim, avg_psnr

    def save_source_parameters(self, flux_gt,flux_pred1,sources_gt, sources_pred, filename):
        """Save the main parameters (x, y, a, b) of the source detection for a single sample to the file"""
        import json
        filename = filename.split('.')[0]
        save_dir = os.path.join('./', 'swimir') 
        os.makedirs(save_dir, exist_ok=True)
        gt_sources = []
        pred_sources = []
        for i in range(len(sources_gt)):
            gt_source = {
                'x': float(sources_gt['x'][i]),
                'y': float(sources_gt['y'][i]),
                'a': float(sources_gt['a'][i]),
                'b': float(sources_gt['b'][i]),
                'flux': float(flux_gt[i])
            }
            gt_sources.append(gt_source)
        
        for i in range(len(sources_pred)):
            pred_source = {
                'x': float(sources_pred['x'][i]),
                'y': float(sources_pred['y'][i]),
                'a': float(sources_pred['a'][i]),
                'b': float(sources_pred['b'][i]),
                'flux': float(flux_pred1[i])
            }
            pred_sources.append(pred_source)

        save_data = {
            'filename': filename,
            'ground_truth': gt_sources,
            'prediction': pred_sources
        }
        
        base_name = os.path.splitext(os.path.basename(filename))[0]
        json_save_path = os.path.join(save_dir, f'{base_name}.json')
        with open(json_save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        npy_save_path = os.path.join(save_dir, f'ours_{base_name}.npy')
        np.save(npy_save_path, save_data)

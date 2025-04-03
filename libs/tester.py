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

class Tester(object):
    def __init__(self, 
                 model, 
                 evalloader, 
                 local_rank=0,
                 ddp=False,
                 visualize=True,
                 vis_dir=None):
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
            idx=1
            pred = results['pred_img'][idx].numpy()  
            target = datalist['hr'][idx].numpy()     
            input_img = datalist['input'][idx].numpy()  
            name = datalist['filename'][idx]           
            vis_astro_SR(pred, target, input_img, name, self.vis_dir)
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
            
        if self.visualize and self.local_rank == 0:
            print("可视化")
            num_samples = len(datalist['hr'])
            indices = random.sample(range(num_samples), min(10, num_samples))
            for idx in indices:
                pred = results['pred_img'][idx].numpy()  
                target = datalist['hr'][idx].numpy()     
                input_img = datalist['input'][idx].numpy()  
                name = datalist['filename'][idx]           
                vis_astro_SR(pred, target, input_img, name, self.vis_dir)


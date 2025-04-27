import pdb
import os
import torch
import sys
import time
import tqdm
from matplotlib.colors import Normalize
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
from .tester import Tester

class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 trainloader,
                 evalloader,
                 logger,
                 max_epoch,
                 log_dir,
                 grad_clip=None,
                 local_rank=0,
                 ddp=False,
                 save_ckp_epoch=10,
                 eval_epoch=10,
                 display_iter=5):
        self.model = model
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epoch = max_epoch
        self.trainloader = trainloader
        self.evalloader = evalloader
        self.log_dir = log_dir
        self.save_ckp_epoch = save_ckp_epoch
        self.display_iter = display_iter
        self.eval_epoch = eval_epoch
        self.epoch = 0
        self.grad_clip = grad_clip
        self.ddp = ddp
        self.rank = local_rank
        logger.info(self.model)
        logger.info(self.optimizer)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if self.ddp:
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[local_rank % torch.cuda.device_count()],
                                                             find_unused_parameters=False)


    def train(self):
        for epoch in range(self.max_epoch):

            self.epoch = epoch + 1
            if self.ddp:
                dist.barrier()
                self.trainloader.sampler.set_epoch(epoch)
            self.train_one_epoch()
            if epoch % self.eval_epoch == 0:
                self.eval_one_epoch()

            if epoch % self.save_ckp_epoch == 0 and self.rank == 0:
                if self.ddp:
                    torch.save(self.model.module.state_dict(), \
                               os.path.join(self.log_dir, 'epoch_{}.pth'.format(self.epoch)))
                else:
                    torch.save(self.model.state_dict(), \
                               os.path.join(self.log_dir, 'epoch_{}.pth'.format(self.epoch)))


    def train_one_epoch(self):
        self.model.train()
        for i, datalist in enumerate(self.trainloader):
            for key in datalist.keys():
                if type(datalist[key]) is torch.Tensor:
                    datalist[key] = datalist[key].to('cuda')

            total_loss, losses = self.model(datalist['input'],datalist)
            self.optimizer.zero_grad()
            total_loss.backward()
            #for p in self.model.module.discriminator.parameters(): print([p.name,p.requires_grad])
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            if (i+1)%self.display_iter==0:
                display_string = "Epoch {:d} [{:d}/{:d}] (lr: {:.6f}): ".format\
                    (self.epoch, i+1, len(self.trainloader), self.optimizer.param_groups[0]['lr'])
                for key in losses.keys():
                    display_string += "{}: {:.6f}, ".format(key,losses[key].item())
                display_string = display_string[:-2]
                if self.grad_clip is not None:
                    total_norm = 0.0
                    for p in self.model.parameters():
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    display_string += ', grad_norm: %.3f'%total_norm
                self.logger.info(display_string)


    def eval_one_epoch(self):
        Tester(self.model, self.evalloader, ddp=self.ddp,logger=self.logger).eval()
        # Tester(self.model, self.evalloader, ddp=self.ddp,logger=self.logger, visualize=True, vis_dir='/home/bingxing2/ailab/scxlab0061/Astro_SR/vis/eval_result_swinIR_l2').eval()

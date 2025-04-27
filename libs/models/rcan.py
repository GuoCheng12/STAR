import time
import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers



import time

from . import MODEL
from .base_model import Base_Model
from .model_init import *



class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)

@MODEL.register
class RCAN(Base_Model):
    def __init__(self, scale,num_features,num_rg,num_rcab,reduction,**kwargs):
        super(RCAN, self).__init__(**kwargs)
        self.scale = scale
        self.num_features = num_features
        self.num_rg = num_rg
        self.num_rcab = num_rcab
        self.reduction = reduction

        self.sf = nn.Conv2d(1, self.num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(self.num_features, self.num_rcab, self.reduction) for _ in range(self.num_rg)])
        self.conv1 = nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features * (self.scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(self.scale)
        )
        self.conv2 = nn.Conv2d(self.num_features, 1, kernel_size=3, padding=1)

    def forward(self, x,targets):
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.upscale(x)
        x = self.conv2(x)
        # return x
        if self.training:
            losses = dict(l1_loss = (torch.abs(x - targets['hr'])*targets['mask']).sum()/(targets['mask'].sum() + 1e-3))
            # losses = dict(l2_loss = ((x - targets['hr'])**2 * targets['mask']).sum() / (targets['mask'].sum() + 1e-3))
            total_loss = torch.stack([*losses.values()]).sum()
            return total_loss, losses
        else:
            return dict(pred_img = x)
if __name__ == '__main__':

    model = RCAN(scale=2,num_features=64, num_rg=10,num_rcab=20, reduction=16)
    print(model)
    # print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 1, 128, 128))
    x = model(x)
    print(x.shape)
    '''
    python main.py --scale 2 \
               --num_rg 10 \
               --num_rcab 20 \ 
               --num_features 64 \              
               --images_dir "" \
               --outputs_dir "" \               
               --patch_size 48 \
               --batch_size 16 \
               --num_epochs 20 \
               --lr 1e-4 \
               --threads 8 \
               --seed 123 \
               --use_fast_loader        
    '''
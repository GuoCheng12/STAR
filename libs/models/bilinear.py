## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers


from einops.layers.torch import Rearrange
import time
from einops import rearrange
from . import MODEL
from .base_model import Base_Model
from .model_init import *
from einops import repeat, rearrange
import torch.nn.functional as F
##########################################################################




##########################################################################
##---------- Bilinear -----------------------
@MODEL.register
class Bilinear(Base_Model):
    def __init__(self, 
        inp_channels=3, 
        scale_factor=2, 
        mode = 'bicubic',
        **kwargs
    ):

        super(Bilinear, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.mode =mode
    def forward(self, inp_img,targets=None):

    
        out_dec_level1 = F.interpolate(inp_img, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)

        return dict(pred_img = out_dec_level1)

        

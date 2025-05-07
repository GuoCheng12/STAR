from .common import *
# import common
import time
import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers


from einops.layers.torch import Rearrange
import time
from . import MODEL
from .base_model import Base_Model
from .model_init import *
from einops import repeat, rearrange
import torch.nn.functional as F

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

# def make_model(args, parent=False):
#     return EDSR(args)

@MODEL.register
class EDSR(Base_Model):
    def __init__(self,  
                
                n_resblocks=32,
                n_feats =64,
                scale= 2,
                res_scale = 0.1,
                n_colors=1,
                rgb_range = 255,
                **kwargs):
        super(EDSR, self).__init__(**kwargs)

        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        kernel_size = 3 
        self.scale = scale
        self.res_scale = res_scale
        self.n_colors = n_colors
        self.rgb_range = rgb_range
        conv=default_conv

        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(self.n_resblocks, self.n_feats, self.scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(self.rgb_range)
        self.add_mean = MeanShift(self.rgb_range, sign=1)

        # define head module
        m_head = [conv(self.n_colors, self.n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, self.n_feats, kernel_size, act=act, res_scale=self.res_scale
            ) for _ in range(self.n_resblocks)
        ]
        m_body.append(conv(self.n_feats, self.n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, self.scale, self.n_feats, act=False),
            conv(self.n_feats, self.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x,targets):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        pred_img = x
        # x = self.add_mean(x)
        if self.training:
            # 提取 targets 中的数据
            attn_map = targets['attn_map']
            mask_float = targets['mask']
            attn_map = torch.nan_to_num(attn_map, nan=0.0)
            # 计算 L1 损失
            l1_loss = (torch.abs(pred_img - targets['hr']) * mask_float).sum() / (mask_float.sum() + 1e-3)
            weighted_diff = torch.abs(pred_img - targets['hr']) * attn_map
            flux_loss = weighted_diff.sum() / (attn_map.sum() + 1e-3)
            total_loss = l1_loss + 0.01 * flux_loss
            losses = dict(l1_loss=l1_loss, flux_loss=0.01*flux_loss)
            return total_loss, losses
        else:
            return dict(pred_img = pred_img)
        # return x 


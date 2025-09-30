
import time
import torch
# print(torch.__version__)
from torch.nn.utils import spectral_norm
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from  basicsr.archs.rrdbnet_arch import RRDBNet 
from einops.layers.torch import Rearrange
import time
from einops import rearrange
from . import MODEL
from .base_model import Base_Model
from .model_init import *
from einops import repeat, rearrange
import torch.nn.functional as F
from basicsr.losses.gan_loss import GANLoss
from .PerceptualLoss import PerceptualLoss



class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out

@MODEL.register
class ESRGAN(Base_Model):
    def __init__(self,  
                
                num_in_ch=1,
                num_feat =64,
                num_out_ch=1,
                num_block=23,
                use_loss='L1',num_grow_ch=32,scale=2,
                use_attention=False,
                stage = 'one',
                device='cuda',
                checkpoints=None,
                **kwargs):
        super(ESRGAN, self).__init__(**kwargs)
        self.checkpoints = checkpoints
        self.use_loss = use_loss
        self.use_attention = use_attention
        self.num_in_ch = num_in_ch
        self.num_feat = num_feat
        self.num_block = num_block
        self.num_grow_ch = num_grow_ch
        self.scale = scale
        self.stage = stage
        self.net_g =  RRDBNet(num_in_ch=self.num_in_ch,num_out_ch=num_out_ch,num_feat=self.num_feat,num_block=self.num_block,num_grow_ch=self.num_grow_ch,scale=self.scale).to(device)
        if self.stage =='two':
            state_dict = torch.load(self.checkpoints)
            print('load checkpoint!!!!')
            self.net_g.load_state_dict(state_dict,strict=False)
        self.net_d = UNetDiscriminatorSN(self.num_in_ch, self.num_feat).to(device)
    def forward(self, x,targets=None):
        import pdb
        # pdb.set_trace()
        # x = self.sub_mean(x)
        output = self.net_g(x)
        # print(x.shape)
        if self.stage =='two':
            real_d_pred = self.net_d(output)
        if self.stage =='one':
            real_d_pred=output
        if self.training:
            mask = targets['mask']
            mask_sum = mask.sum() + 1e-3
            if self.use_loss == 'L1':
                base_loss = (torch.abs(real_d_pred - targets['hr']) * mask).sum() / mask_sum
                loss_name = 'l1_loss'
            elif self.use_loss == 'L2':
                base_loss = ((real_d_pred - targets['hr'])**2 * mask).sum() / mask_sum
                loss_name = 'l2_loss'

            losses = {loss_name: base_loss}
            total_loss = base_loss

            
            if self.stage=='two':
                cri_gan = GANLoss(gan_type='vanilla',real_label_val=1.0,fake_label_val=0,loss_weight=0.1)
                l_g_gan = cri_gan(real_d_pred, True, is_disc=False)
                losses['gan_loss'] = l_g_gan
                # pdb.set_trace()
            
                # PelLoss = PerceptualLoss(layer_weights= 
                #                             {'conv1_2':0.1,
                #                             'conv2_2': 0.1,
                #                             'conv3_4': 1,
                #                             'conv4_4': 1,
                #                             'conv5_4': 1},
                #                             vgg_type= 'vgg19',
                #                             use_input_norm= True,
                #                             perceptual_weight= 1.0,
                #                             style_weight=0,
                #                             range_norm=False,
                #                             criterion='l1').cuda()
                # l_g_percep, l_g_style = PelLoss(real_d_pred* mask, targets['hr']* mask)
                total_loss = base_loss+l_g_gan
                # if l_g_percep is not None:
                #     losses['l_g_percep'] = l_g_percep
                #     total_loss = total_loss+l_g_percep
                # if l_g_style is not None:
                #     losses['l_g_style'] = l_g_style
                #     total_loss = total_loss+l_g_style

            

            if self.use_attention:
                attn_map = torch.nan_to_num(targets['attn_map'], nan=0.0)
                weighted_diff = torch.abs(real_d_pred - targets['hr']) * attn_map
                flux_loss = weighted_diff.sum() / (attn_map.sum() + 1e-3)
                losses['flux_loss'] = 0.01 * flux_loss
                total_loss = total_loss + 0.01 * flux_loss
            return total_loss, losses
        else:
            return dict(pred_img = real_d_pred)
        # return x 
if __name__ == '__main__':
    upscale = 4
    from  basicsr.archs.rrdbnet_arch import RRDBNet 
    model1 = RRDBNet(num_in_ch=1,num_out_ch=1,num_feat=64,num_block=23,num_grow_ch=32,scale=2)
    model = ESRGAN()
    # print(model)
    

    x = torch.randn((1,1, 128, 128))
    y = torch.randn((1,1, 256, 256))
    total_loss, losses= model(x,y)
    print(total_loss, losses)


#     network_g:
#   type: RRDBNet
#   num_in_ch: 3
#   num_out_ch: 3
#   num_feat: 64
#   num_block: 23
#   num_grow_ch: 32
#   scale: 2
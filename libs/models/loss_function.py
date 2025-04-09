import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F

class general_loss(nn.Module):
    def __init__(self, weight, loss_function):
        super(general_loss, self).__init__()
        self.weight=weight
        self.loss_function=loss_function
    def forward(self, inputs):
        loss = self.loss_function(*inputs)
        return self.weight*loss

class curriculum_focal_loss_heatmap(nn.Module):
    def __init__(self, alpha=2.0, gamma=4.0):
        super(curriculum_focal_loss_heatmap, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.iter_num = 0
    def forward(self, input, target):
        '''
        Args:
            input:  prediction, 'batch x c x h x w'
            target:  ground truth, 'batch x c x h x w'
            alpha: hyper param, default in 0.25
            gamma: hyper param, default in 2.0
        Reference: Focal Loss for Dense Object Detection, ICCV'17
        '''
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(input + eps).log() * (1 - input).pow(self.alpha) * pos_weights
        neg_loss = -(1 - input + eps).log() * input.pow(self.alpha) * neg_weights
        self.iter_num += 1
        return (pos_loss + neg_loss*10**min(0,self.iter_num//(7*100)-6)).sum()/pos_weights.sum()
        #return (pos_loss + neg_loss).sum()/pos_weights.sum()

class Smooth_L1(nn.Module):
    def __init__(self):
        super(Smooth_L1, self).__init__()
    def forward(self, inputs, targets):
        loss_function = nn.SmoothL1Loss()
        return loss_function(inputs, targets)

class L1_loss(nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
    def forward(self, inputs, targets):
        loss_function = nn.L1Loss()
        return loss_function(inputs, targets)
        
class MSE_loss(nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()
    def forward(self, inputs, targets):
        loss_function = nn.MSELoss()
        return loss_function(inputs, targets)
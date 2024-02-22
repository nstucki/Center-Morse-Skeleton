import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append("../previous_skeleton")
from previous_skeleton import MentenSkeletonize, ShitSkeletonize, VitiSkeletonize
from morse_skeleton import DMTSkeletonize


class soft_cldice(nn.Module):
    def __init__(self, mode='Shit', iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        if mode == 'DMT':
            self.skeletonization_module = DMTSkeletonize(reparameterize='fixed')
        if mode == 'Menten': 
            self.skeletonization_module = MentenSkeletonize(num_iter=10)
        if mode == 'Viti':
            self.skeletonization_module = VitiSkeletonize(num_iter=10)
        if mode == 'Shit':
            self.skeletonization_module = ShitSkeletonize(num_iter=10)

    def forward(self, y_true, y_pred):
        skel_pred = self.skeletonization_module(y_pred)
        skel_true = self.skeletonization_module(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice, {}


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, mode='Shit', iter_=3, alpha=0.5, smooth = 1.):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        if mode == 'DMT':
            self.skeletonization_module = DMTSkeletonize(reparameterize='fixed')
        if mode == 'Menten': 
            self.skeletonization_module = MentenSkeletonize(num_iter=10)
        if mode == 'Viti':
            self.skeletonization_module = VitiSkeletonize(num_iter=10)
        if mode == 'Shit':
            self.skeletonization_module = ShitSkeletonize(num_iter=10)

    def forward(self, y_true, y_pred):
        dice = soft_dice(y_true, y_pred)
        skel_pred = self.skeletonization_module(y_pred)
        skel_true = self.skeletonization_module(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice, {'dice':dice}

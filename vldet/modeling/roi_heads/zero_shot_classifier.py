# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec

class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        detection_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0, 
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        share_proj_v_dim = 1024,
        share_proj_l_dim = 1024,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(share_proj_v_dim, share_proj_l_dim)

        #zs head
        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(
                torch.load(zs_weight_path), 
                dtype=torch.float32).permute(1, 0).contiguous() # D x C

        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))], 
            dim=1) # D x (C + 1)

        # detection head
        if detection_weight_path == 'rand':
            detection_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            detection_weight = torch.tensor(
                torch.load(detection_weight_path), 
                dtype=torch.float32).permute(1, 0).contiguous() # D x C
                
        detection_weight = torch.cat(
            [detection_weight, detection_weight.new_zeros((zs_weight_dim, 1))], 
            dim=1) # D x (C + 1)
        
        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
            detection_weight = F.normalize(detection_weight, p=2, dim=0)
        
        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)
        
        if detection_weight_path == 'rand':
            self.detection_weight = nn.Parameter(detection_weight)
        else:
            self.register_buffer('detection_weight', detection_weight)

        assert self.detection_weight.shape[1] == num_classes + 1, self.detection_weight.shape


    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'detection_weight_path': cfg.MODEL.ROI_BOX_HEAD.DETECTION_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            'share_proj_v_dim': cfg.MODEL.SHARE_PROJ_V_DIM,
            'share_proj_l_dim': cfg.MODEL.SHARE_PROJ_L_DIM,
        }

    def forward(self, input_x, ann_type='box', classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        proj_x = self.linear(input_x)


        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous() # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            if self.training and ann_type != 'box':
                zs_weight = self.zs_weight
            else:
                zs_weight = self.detection_weight

        if self.norm_weight:
            proj_x = self.norm_temperature * F.normalize(proj_x, p=2, dim=1)
        
        x = torch.mm(proj_x, zs_weight)

        if self.use_bias:
            x = x + self.cls_bias
            
        return x, proj_x, zs_weight
        
class SinkhornDistance(torch.nn.Module):
    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

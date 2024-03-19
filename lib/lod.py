# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F, Conv2d, ConvTranspose2d
from typing import List
import cv2
import torchvision.transforms as transforms

import math
class LODBorderHead(nn.Module):
    """
    A head for Oriented derivatives learning, adaptive thresholding and feature fusion.
    """

    def __init__(self,input_channels):

        super(LODBorderHead, self).__init__()
       # self.input_channels = input_shape.channels
        self.input_channels =input_channels
        self.num_directions =8
        self.num_classes = 1

        self.conv_norm_relus = []

        cur_channels = 256

        self.layers = nn.Sequential(
            Conv2d(
                self.input_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            ConvTranspose2d(
                cur_channels, cur_channels, kernel_size=2, stride=2, padding=0
            ),
            nn.ReLU(),
            Conv2d(cur_channels, self.num_directions, kernel_size=3, padding=1, stride=2),
        )

        self.offsets_conv = nn.Sequential(
            Conv2d(
                self.num_directions,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            ConvTranspose2d(
                cur_channels, cur_channels, kernel_size=2, stride=2, padding=0
            ),
            ConvTranspose2d(
                cur_channels, cur_channels, kernel_size=2, stride=2, padding=0
            ),
            nn.ReLU(),
            Conv2d(cur_channels, self.num_directions * 2, kernel_size=3, padding=1, stride=2),
            Conv2d(self.num_directions*2, 1, kernel_size=3, padding=1, stride=2),
        )

        self.predictor = Conv2d(64, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.outputs_conv = Conv2d(8, input_channels, kernel_size=1)
        self.fusion_conv = nn.Sequential(
            Conv2d(
                self.input_channels + 1,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_msra_fill(m)
            if isinstance(m, ConvTranspose2d):
                weight_init.c2_msra_fill(m)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # use normal distribution initialization for mask prediction layer
        weight_init.c2_msra_fill(self.outputs_conv)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, features,mask_logits,ATT):
        points_num=50
        pred_od = self.layers(features)
        #pred_od= F.interpolate(pred_od, scale_factor=0.25, mode='bilinear') 
        N,C,W,H=pred_od.shape
        sum_od=torch.zeros([N,1,W,H],device=pred_od.device)
        for i in range(N):
          for num in range(8):
            sum_od[i,:,:,:]=sum_od[i,:,:,:]+pred_od[i,num,:,:].abs()  
        ATT_od=ATT*sum_od
        #od_activated_map = self.adaptive_thresholding(points_num, ATT_od)
        border_mask_logits = self.boundary_aware_mask_scoring(mask_logits, ATT_od, pred_od)
        return border_mask_logits,pred_od


    def boundary_aware_mask_scoring(self, mask_logits, od_activated_map, pred_od):
        od_activated_map=od_activated_map.abs()
        od_activated_map=(od_activated_map-od_activated_map.min())/(od_activated_map.max()-od_activated_map.min()+1e-10)
        od_features = self.outputs_conv(pred_od)
        #od_activated_map = od_activated_map.unsqueeze(dim=1)
        mask_fusion_scores = mask_logits + od_features
        border_mask_scores = (1-od_activated_map) * mask_logits + od_activated_map * mask_fusion_scores
        return border_mask_scores


    def adaptive_thresholding(self, points_num, od_features):
        od_features=od_features.abs()
        N, C, H, W = od_features.shape
        
        res=(od_features-od_features.min())/(od_features.max()-od_features.min()+1e-10)
        activated_map=res>0.1
       # od_features=od_features.view(N,H*W)
        #_, idxx = torch.topk(od_features, points_num)
        #shift = H * W * torch.arange(N, dtype=torch.long, device=idxx.device)
        #idxx += shift[:, None]
        #activated_map = torch.zeros([N, H * W], dtype=torch.bool, device=od_features.device)
        #activated_map.view(-1)[idxx.view(-1)] = True
    
        return activated_map.view(N, H, W)

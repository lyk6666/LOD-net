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

    def __init__(self, input_channels):

        super(LODBorderHead, self).__init__()
        # self.input_channels = input_shape.channels
        self.input_channels = input_channels
        self.num_directions = 8
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
            Conv2d(self.num_directions * 2, 1, kernel_size=3, padding=1, stride=2),
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

    def forward(self, features, mask_logits, challenging_to_discriminate_region_map):
        pred_od = self.layers(features)
        N, C, W, H = pred_od.shape
        proposal_boundary_region_map = torch.zeros([N, 1, W, H], device=pred_od.device)
        for i in range(N):
            for num in range(8):
                proposal_boundary_region_map[i, :, :, :] =  proposal_boundary_region_map[i, :, :, :] + pred_od[i, num, :, :].abs()
        Hard_region_aware_map = challenging_to_discriminate_region_map *  proposal_boundary_region_map
        border_mask_logits = self.boundary_aware_mask_scoring(mask_logits, Hard_region_aware_map, pred_od)
        return border_mask_logits, pred_od

    def boundary_aware_mask_scoring(self, mask_logits, Hard_region_aware_map, pred_od):
        Hard_region_aware_map = Hard_region_aware_map.abs()
        Hard_region_aware_map = (Hard_region_aware_map - Hard_region_aware_map.min()) / (
                Hard_region_aware_map.max() - Hard_region_aware_map.min() + 1e-10)
        od_features = self.outputs_conv(pred_od)
        mask_fusion_scores = mask_logits + od_features
        border_mask_scores = (1 - Hard_region_aware_map) * mask_logits + Hard_region_aware_map * mask_fusion_scores
        return border_mask_scores


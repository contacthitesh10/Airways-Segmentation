import cv2
import os
import torch
import tqdm
import random
import torchvision
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchsummary import summary
import torchvision.transforms as trf
import torchvision.transforms.functional as fn

class UNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, up = False):
        super(UNetBlock, self).__init__()

        padding = {'padding': 1}
        self.up = up
        layer_name = 'dec' if up else 'enc'


        self.conv_layer = nn.Sequential(OrderedDict([
            (f'{layer_name}_c1', nn.Conv2d(in_channels, out_channels, kernel_size = 
3, stride = 1, **padding)),
            (f'{layer_name}_bn1', nn.BatchNorm2d(out_channels)),
            (f'{layer_name}_a1', nn.ReLU()),
            (f'{layer_name}_c2', nn.Conv2d(out_channels, out_channels, kernel_size = 
3, stride = 1, **padding)),
            (f'{layer_name}_bn1', nn.BatchNorm2d(out_channels)),
            (f'{layer_name}_a2', nn.ReLU()),
        ]))
        self.l0 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, 
stride = 2) if up else nn.MaxPool2d(kernel_size = 2)

    def forward(self, X, r_X = None):
        if self.up:
            ups_out = self.l0(X)
            crop_to_shape = ups_out.shape[2:]
            r_X = torchvision.transforms.CenterCrop(crop_to_shape)(r_X)

            ups_out = torch.cat([ups_out, r_X], axis = 1)

            return self.conv_layer(ups_out)
        else:
            out_conv = self.conv_layer(X)
            out_pooling = self.l0(out_conv)
            return out_pooling, out_conv

class SkinDiseaseSGv1(torch.nn.Module):
    def __init__(self, in_channels = 3):
        super(SkinDiseaseSGv1, self).__init__()

        self.enc_l0 = UNetBlock(3, 64)
        self.enc_l1 = UNetBlock(64, 128)
        self.enc_l2 = UNetBlock(128, 256)
        self.enc_l3 = UNetBlock(256, 512)

        self.bn = nn.Sequential(OrderedDict([
            ('bn_c1', nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 
1)),
            ('enc_bn1', nn.BatchNorm2d(1024)),
            ('bn_a1', nn.ReLU()),
            ('bn_c2', nn.Conv2d(1024, 1024, kernel_size = 3, stride = 1, padding = 
1)),
            ('enc_bn1', nn.BatchNorm2d(1024)),
            ('bn_a2', nn.ReLU()),
        ]))

        self.dec_l0 = UNetBlock(1024, 512, up = True)
        self.dec_l1 = UNetBlock(512, 256, up = True)
        self.dec_l2 = UNetBlock(256, 128, up = True)
        self.dec_l3 = UNetBlock(128, 64, up = True)

        self.clf = nn.Sequential(OrderedDict([
            ('oc1', nn.Conv2d(64, 1, kernel_size = 1, stride = 1)),
#           ('oa1', nn.Sigmoid())
        ]))


    def forward(self, X):

        # encoding
        enc_l0_out, X_r0 = self.enc_l0(X)
        enc_l1_out, X_r1 = self.enc_l1(enc_l0_out)
        enc_l2_out, X_r2 = self.enc_l2(enc_l1_out)
        enc_l3_out, X_r3 = self.enc_l3(enc_l2_out)

        # bottleneck
        bn_out = self.bn(enc_l3_out)

        # decoding
        dec_l0_out = self.dec_l0(bn_out, X_r3)
        dec_l1_out = self.dec_l1(dec_l0_out, X_r2)
        dec_l2_out = self.dec_l2(dec_l1_out, X_r1)
        dec_l3_out = self.dec_l3(dec_l2_out, X_r0)

        return self.clf(dec_l3_out)

lr_model = SkinDiseaseSGv1()

import os
import numpy as np
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from typing import Callable, Union

## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=input_nc, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x
############################################################## 3D UNET ##########################################################################
## 네트워크 구축하기
class UNet_Encoder_3D(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(UNet_Encoder_3D, self).__init__()

        def CBR3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm3d(num_features=out_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)
            return cbr

        # Contracting path
        self.enc1_1 = CBR3d(in_channels=input_nc, out_channels=64)
        self.enc1_2 = CBR3d(in_channels=64, out_channels=64)
        self.pool1  = nn.MaxPool3d(kernel_size=2)

        self.enc2_1 = CBR3d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR3d(in_channels=128, out_channels=128)
        self.pool2  = nn.MaxPool3d(kernel_size=2)

        self.enc3_1 = CBR3d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR3d(in_channels=256, out_channels=256)
        self.pool3  = nn.MaxPool3d(kernel_size=2)

        self.enc4_1 = CBR3d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR3d(in_channels=512, out_channels=512)
        self.pool4  = nn.MaxPool3d(kernel_size=2)

        self.enc5_1 = CBR3d(in_channels=512, out_channels=1024)


    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        return enc5_1, enc4_2, enc3_2, enc2_2, enc1_2



## 네트워크 구축하기
class UNet_Decoder_3D(nn.Module):
    def __init__(self):
        super(UNet_Decoder_3D, self).__init__()

        def CBR3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm3d(num_features=out_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)
            return cbr


        # Expansive path
        self.dec5_1 = CBR3d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR3d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR3d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR3d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR3d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR3d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR3d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR3d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR3d(in_channels=64, out_channels=64)


    def forward(self, enc5_1, enc4_2, enc3_2, enc2_2, enc1_2):

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        return dec1_1



########################################################################### 2D Unet #########################################################################################
## 네트워크 구축하기
class UNet_Encoder(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(UNet_Encoder, self).__init__()

        def CBR3d(in_channels, out_channels, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True):
            layers = []
            layers += [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm3d(num_features=out_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)
            return cbr

        # Contracting path
        self.enc1_1 = CBR3d(in_channels=input_nc, out_channels=64)
        self.enc1_2 = CBR3d(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.enc2_1 = CBR3d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR3d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.enc3_1 = CBR3d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR3d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.enc4_1 = CBR3d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR3d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.enc5_1 = CBR3d(in_channels=512, out_channels=1024)


    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        return enc5_1, enc4_2, enc3_2, enc2_2, enc1_2



## 네트워크 구축하기
class UNet_Decoder(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(UNet_Decoder, self).__init__()

        def CBR3d(in_channels, out_channels, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True):
            layers = []
            layers += [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm3d(num_features=out_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)
            return cbr


        # Expansive path
        self.dec5_1 = CBR3d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0, bias=True)

        self.dec4_2 = CBR3d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR3d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0, bias=True)

        self.dec3_2 = CBR3d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR3d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0, bias=True)

        self.dec2_2 = CBR3d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR3d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0, bias=True)

        self.dec1_2 = CBR3d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR3d(in_channels=64, out_channels=64)


    def forward(self, enc5_1, enc4_2, enc3_2, enc2_2, enc1_2):

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        return dec1_1

########################################### SEG HEAD ############################################

class Seg_Head(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(Seg_Head, self).__init__()

        # Expansive path
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True)
        self.bn1   = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True)
        self.bn2   = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()

        self.last  = nn.Conv3d(in_channels=64, out_channels=output_nc, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True)

    def forward(self, x_skip):

        x = self.conv1(x_skip)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.last(x)

        return x



class Seg_Head_Conv3d(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(Seg_Head_Conv3d, self).__init__()

        # Expansive path
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1   = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2   = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()

        self.last  = nn.Conv3d(in_channels=64, out_channels=output_nc, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x_skip):

        x = self.conv1(x_skip)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.last(x)

        return x







########### CLS HEAD ####################


class PyramidPooling(nn.Module):

    def __init__(self, pooling: Callable, levels: int = 1):
        super().__init__()
        self.levels = levels
        self.pooling = pooling

    def forward(self, x):
        assert x.dim() > 2
        shape = np.array(x.shape[2:], dtype=int)
        batch_size = x.shape[0]
        pyramid = []

        for level in range(self.levels):
            # adaptive pooling
            level = 2 ** level
            stride = np.floor(shape / level)
            kernel_size = shape - (level - 1) * stride
            stride, kernel_size = tuple(map(int, stride)), tuple(map(int, kernel_size))
            temp = self.pooling(x, kernel_size=kernel_size, stride=stride)
            # print("!!! = ", temp.shape)   # torch.Size([2, 1024, 1, 1, 1])
            pyramid.append(temp.view(batch_size, -1))

        return torch.cat(pyramid, dim=-1)

    @staticmethod
    def get_multiplier(levels, ndim):
        return (2 ** (ndim * levels) - 1) // (2 ** ndim - 1)

class Cls_Head(nn.Module):
    def __init__(self, levels=4):
        super(Cls_Head, self).__init__()

        # Expansive path
        self.pyramid_pool  = PyramidPooling(partial(F.max_pool3d, ceil_mode=True), levels=levels)
        self.dropout = nn.Dropout()
        # self.linear  = nn.Linear(PyramidPooling.get_multiplier(levels=4, ndim=3) * 8, 1024)
        self.linear  = nn.Linear(PyramidPooling.get_multiplier(levels=levels, ndim=3)*1024, 1024)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout()
        self.last    = nn.Linear(1024, 1)  # shape (N, 1)

    def forward(self, x):

        x = self.pyramid_pool(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.last(x)

        return x

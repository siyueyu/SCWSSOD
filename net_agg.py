#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out      = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out      = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out      = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out+residual, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes*4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('resnet50-19c8e357.pth'), strict=False)

class CA(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(CA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256
        down = down.mean(dim=(2,3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        return left * down

    def initialize(self):
        weight_init(self)

""" Self Refinement Module """
class SRM(nn.Module):
    def __init__(self, in_channel):
        super(SRM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]
        return F.relu(w * out1 + b, inplace=True)


    def initialize(self):
        weight_init(self)


""" Feature Interweaved Aggregation Module """
class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(FAM, self).__init__()
        #self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_right, 256, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv_att1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.conv_att2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.conv_att3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True) #256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True) #256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear', align_corners=False)
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)
        z1_att = F.adaptive_avg_pool2d(self.conv_att1(z1), (1,1))
        z1 = z1_att * z1

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear', align_corners=False)

        z2 = F.relu(down_1 * left, inplace=True)
        z2_att = F.adaptive_avg_pool2d(self.conv_att2(z2), (1,1))
        z2 = z2_att * z2

# z3
        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear', align_corners=False)
        z3 = F.relu(down_2 * left, inplace=True)
        z3_att = F.adaptive_avg_pool2d(self.conv_att3(z3), (1,1))
        z3 = z3_att * z3
        out = (z1 + z2 + z3) / (z1_att + z2_att + z3_att)
        # out = torch.cat((z1, z2, z3), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)


    def initialize(self):
        weight_init(self)


class SA(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(SA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_down, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256 channels
        down_1 = self.conv2(down) #wb
        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear', align_corners=False)
        w,b = down_1[:,:256,:,:], down_1[:,256:,:,:]

        return F.relu(w*left+b, inplace=True)

    def initialize(self):
        weight_init(self)


class SCWSSOD(nn.Module):
    def __init__(self, cfg):
        super(SCWSSOD, self).__init__()
        self.cfg     = cfg
        self.bkbone  = ResNet()

        self.ca45    = CA(2048, 2048)
        self.ca35    = CA(2048, 2048)
        self.ca25    = CA(2048, 2048)
        self.ca55    = CA(256, 2048)
        self.sa55   = SA(2048, 2048)

        self.fam45   = FAM(1024,  256, 256)
        self.fam34   = FAM( 512,  256, 256)
        self.fam23   = FAM( 256,  256, 256)

        self.srm5    = SRM(256)
        self.srm4    = SRM(256)
        self.srm3    = SRM(256)
        self.srm2    = SRM(256)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x, mode=None):
        out1, out2, out3, out4, out5_ = self.bkbone(x)
        # GCF
        out4_a = self.ca45(out5_, out5_)
        out3_a = self.ca35(out5_, out5_)
        out2_a = self.ca25(out5_, out5_)
        # HA
        out5_a = self.sa55(out5_, out5_)
        out5 = self.ca55(out5_a, out5_)
        # out
        out5 = self.srm5(out5)
        out4 = self.srm4(self.fam45(out4, out5, out4_a))
        out3 = self.srm3(self.fam34(out3, out4, out3_a))
        out2 = self.srm2(self.fam23(out2, out3, out2_a))
        # we use bilinear interpolation instead of transpose convolution
        if mode == 'Test':
            # ------------------------------------------------------ TEST ----------------------------------------------------
            out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode='bilinear', align_corners=False)
            out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode='bilinear', align_corners=False)
            out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode='bilinear', align_corners=False)
            out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode='bilinear', align_corners=False)
            return out2, out3, out4, out5
        else:
            # ------------------------------------------------------ TRAIN ----------------------------------------------------
            out5  = torch.sigmoid(F.interpolate(self.linear5(out5), size=x.size()[2:], mode='bilinear', align_corners=False))
            out4  = torch.sigmoid(F.interpolate(self.linear4(out4), size=x.size()[2:], mode='bilinear', align_corners=False))
            out3  = torch.sigmoid(F.interpolate(self.linear3(out3), size=x.size()[2:], mode='bilinear', align_corners=False))
            out2  = torch.sigmoid(F.interpolate(self.linear2(out2), size=x.size()[2:], mode='bilinear', align_corners=False))
            out5 = torch.cat((1 - out5, out5), 1)
            out4 = torch.cat((1 - out4, out4), 1)
            out3 = torch.cat((1 - out3, out3), 1)
            out2 = torch.cat((1 - out2, out2), 1)
            return out2, out3, out4, out5
    def initialize(self):
        weight_init(self)
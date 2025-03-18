import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import os
import cv2
import fnmatch

import pretrainedmodels

class Resnet(nn.Module):
   
    def __init__(self):
        super(Resnet, self).__init__()
        # TODO: check out the se minimax optim sometime
        # bottleneck layers are 1x1 conv, for dim reduction, 
        # then 3x3 conv for feature transform, then upscale with 1x1
        # ref: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006


        backbone = pretrainedmodels.__dict__["resnet50"](num_classes=1000, pretrained="imagenet")

        in_features = backbone.last_linear.in_features
        od = OrderedDict()
        od['conv1'] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        od['bn1'] = nn.BatchNorm2d(64)
        od['relu'] = nn.ReLU(inplace=True)
        od['maxpool'] = nn.MaxPool2d(kernel_size=3,  stride=2, padding = 1) 
 
        self.layer0 = nn.Sequential(od)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def resnet_forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
                
        return x


class SequenceModel(nn.Module):

    def __init__(self):
        super(SequenceModel, self).__init__()

        self.recurrent = nn.LSTM(input_size=2048, hidden_size=512,
            dropout=0.3, num_layers=2,
            bidirectional=True, batch_first=True)
      
        self.fc = nn.Linear(1024, 6)

    def forward(self, x, seq_len):
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = x.reshape(-1, seq_len, x.size(-1))
        x, _ = self.recurrent(x)
        x = self.fc(x)
        x = x.reshape(-1, x.size(-1))
                

        return x


class CaireICH(Resnet):
    def __init__(self):
        super(CaireICH, self).__init__()
        self.decoder = SequenceModel()

    def forward(self, x, seq_len):
        x = self.resnet_forward(x)
        x = self.decoder(x, seq_len)
        return x

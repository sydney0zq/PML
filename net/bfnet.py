#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

"""

"""
import torch
import torch.nn as nn
from collections import OrderedDict
import os
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(CURR_DIR)

from res_deeplab import Res_Deeplab

from net_utils import init_weights

class BFVOSNet(nn.Module):
    def __init__(self, embedding_vector_dims=128, logger=None):
        super().__init__()
        self.base_extractor = Res_Deeplab()
        #self.embedding_head = nn.Sequential(
        #    OrderedDict([
        #        ('conv1', nn.Conv2d(2048+2, embedding_vector_dims, 1, 1)),
        #        ('relu1', nn.ReLU()),
        #        ]))
        #self.embedding_head.add_module('eh_layer2', nn.Conv2d(embedding_vector_dims, embedding_vector_dims, 1, 1))
        #Hobot network
        self.embedding_head = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(2048+2, embedding_vector_dims, kernel_size=3, stride=1, padding=1)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(embedding_vector_dims, embedding_vector_dims, kernel_size=1, stride=1)),
                ('relu2', nn.ReLU())
                ]))
        #('conv2', nn.Conv2d(embedding_vector_dims, embedding_vector_dims, kernel_size=3, stride=1, padding=1)),
        #('relu2', nn.ReLU()),
        #('bn1', nn.BatchNorm2d(embedding_vector_dims)),
        #('bn2', nn.BatchNorm2d(embedding_vector_dims)),
        self.embedding_head.add_module('eh_layer', nn.Conv2d(embedding_vector_dims, embedding_vector_dims, 1, 1))
        self.logger = logger
        init_weights(self)
    
    def forward(self, x, y):
        ## x is image batch; y is 3 channel tensor with (i, j, t) spatio-temporal information
        # x is image batch; y is 2 channel tensor with (i, j) spatio-temporal information
        deeplab_features = self.base_extractor.forward(x)
        embedding = self.embedding_head(torch.cat((deeplab_features, y), dim=1))
        normalized_embedding = embedding / embedding.pow(2).sum(1, keepdim=True).sqrt() # NOTE
        return normalized_embedding
        #return embedding

    def freeze_bn(self, name="base_extractor"):
        if name == "base_extractor":
            for m in self.base_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        elif name == "all":
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        else:
            assert(False), "Your index name nonexist..."

    def check_bn(self, name="base_extractor"):
        if name == "base_extractor":
            for m in self.base_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if m.training is True:
                        self.logger.warning ("Base extractor BN training is ON...")
                        return True
            self.logger.warning ("Base extractor BN training is OFF...")
        elif name == "embedding_head":
            for m in self.embedding_head.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if m.training is True:
                        self.logger.warning ("Embedding head BN training is ON...")
                        return True
            self.logger.warning ("Embedding head BN training is OFF...")
        else:
            assert(False), "Your index name nonexist..."

    def freeze_feature_extraction(self):
        for param in self.base_extractor.parameters():
                param.requires_grad = False

    def get_1x_lr_params(self):
        for m in self.base_extractor.modules():
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    def get_10x_lr_params(self):
        for m in self.embedding_head.modules():
            for p in m.parameters():
                if p.requires_grad:
                    yield p

if __name__ == "__main__":
    model = BFVOSNet()
    model.eval()

    #image = torch.autograd.Variable(torch.randn(1, 3, 854, 480))
    #y = torch.autograd.Variable(torch.randn(1, 2, 107, 61))
    #print(model(image, y)[0].size())
    #params = [x for x in model.get_1x_lr_params()]
    print (params)
    



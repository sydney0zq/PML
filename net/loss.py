#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

"""
Loss function.
"""

import torch
import torch.nn.functional as F

def distance_matrix(x, y):
    """
    Compute distance matrix between two sets of embedding points
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    :x: Nxd tensor
    :y: Mxd tensor
    return NxM distance matrix
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, min=0.0)

class minTripletLoss(torch.nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self._alpha = alpha

    # Anchor points: Axd tensor Pos points: Pxd tensor Neg points: Nxd
    def forward(self, anchor_points, positive_pool, negative_pool):
        positive_distances = distance_matrix(anchor_points, positive_pool)
        negative_distances = distance_matrix(anchor_points, negative_pool)
        losses = F.relu(torch.min(positive_distances, 1)[0]-torch.min(negative_distances, 1)[0]+self._alpha)
        losses = losses.sum() / anchor_points.size(0)
        #losses = losses.sum()
        return losses




if __name__ == "__main__":
    from torch.autograd import Variable, gradcheck

    anchor = torch.randn((2, 3))
    pos = torch.randn((5, 3))
    neg = torch.randn((7, 3))
    anchor_v = Variable(anchor, requires_grad=True)
    pos_v = Variable(pos, requires_grad=True)
    neg_v = Variable(neg, requires_grad=True)

    loss_fn = minTripletLoss().double()
    print (loss_fn(anchor, pos, neg))
    print (loss_fn(anchor_v, pos_v, neg_v))

    print (gradcheck(loss_fn, [anchor.double(), pos.double(), neg.double()]))
    print (gradcheck(loss_fn, [anchor_v.double(), pos_v.double(), neg_v.double()]))

    #loss = loss_fn.forward(anchor, pos, neg)
    import pdb
    pdb.set_trace()











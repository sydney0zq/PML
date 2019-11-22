#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

"""

"""
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import torch

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01) 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01) 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.weight, 1)

def create_triplet_pools(triplet_sample, embeddings, phase='train'):
    """
    Fast vectorized ops to create lots of point-wise triplets from a data loader's sample (3 frames) and embeddings
    :param triplet_sample: dict where sample['image'] is a 3 x 3 W x H tensor
    :param embeddings: 3 x 128 x (W/8) X (H/8) tensor
    :return:
    """
    embedding_a = embeddings[0]
    embedding_f1 = embeddings[1]
    embedding_f2 = embeddings[2]
    embedding_pool_points = torch.cat([embedding_f1, embedding_f2], 2)  # horizontally stacked frame1 and frame 2, 128x33x66
    # embedding_a/p/n is of shape (d, w/8, h/8)

    anchor_points = triplet_sample['annotation'][0]  # all anchor points
    fg_anchor_indices = torch.nonzero(anchor_points)
    bg_anchor_indices = torch.nonzero(anchor_points == 0)

    if fg_anchor_indices.numel() == 0 or bg_anchor_indices.numel() == 0:
        return None

    if phase == 'train':
        n_fg_anchor, n_bg_anchor = 256, 256
        fg_anchor_reindices = torch.randint(0, fg_anchor_indices.size(0), (n_fg_anchor,)).type(torch.LongTensor)
        fg_anchor_indices = fg_anchor_indices[fg_anchor_reindices]
        bg_anchor_reindices = torch.randint(0, bg_anchor_indices.size(0), (n_bg_anchor,)).type(torch.LongTensor)
        bg_anchor_indices = bg_anchor_indices[bg_anchor_reindices]

    
    # all_pool_points is a binary tensor of shape (w/8, h/8).
    # For any index in all_pool_points, if it 1 => it is a foreground pixel
    all_pool_points = torch.cat([triplet_sample['annotation'][1], triplet_sample['annotation'][2]], 1)  # 33x66
    fg_pool_indices = torch.nonzero(all_pool_points)
    bg_pool_indices = torch.nonzero(all_pool_points == 0)

    if fg_pool_indices.numel() == 0 or bg_pool_indices.numel() == 0:
        return None

    fg_embedding_a = torch.cat([embedding_a[:, x, y].unsqueeze(0) for x, y in fg_anchor_indices])
    bg_embedding_a = torch.cat([embedding_a[:, x, y].unsqueeze(0) for x, y in bg_anchor_indices])

    # Compute loss for all foreground anchor points
    # For foreground anchor points,
    # positive pool: all foreground points in all_pool_points
    # negative pool: all background points in all_pool_points
    fg_positive_pool = torch.cat([embedding_pool_points[:, x, y].unsqueeze(0) for x, y in fg_pool_indices])
    bg_positive_pool = torch.cat([embedding_pool_points[:, x, y].unsqueeze(0) for x, y in bg_pool_indices])

    fg_negative_pool = bg_positive_pool
    bg_negative_pool = fg_positive_pool

    return [fg_embedding_a.cuda(), bg_embedding_a.cuda(), \
            fg_positive_pool.cuda(), fg_negative_pool.cuda(), \
            bg_positive_pool.cuda(), bg_negative_pool.cuda()]


"""
Usage:
X is a 128xN tensor embedding
cur_shape  = (height, width)
pca_img = projection_for_visualization(X, 3, cur_shape)
"""


def normalization(data):
    data *= 1.0
    if len(np.unique(data)) == 1:
        return data * 0
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def projection_for_visualization(X, n_components, cur_shape):
    pca = PCA(n_components=n_components, copy=True)
    X_projected = pca.fit_transform(X)
    Z = np.reshape(X_projected, (cur_shape[0], cur_shape[1], n_components))
    pca_img = normalization(Z)
    return pca_img

def visual_embeddings(embeddings, image, summary_writer, idx, embedding_vec_dims=128):
    image_b = image[0:3].data.cpu().numpy()
    image_a, image_p, image_n = image_b[0], image_b[1], image_b[2]
    embedding_b = embeddings[0:3].data.cpu().numpy()
    embedding_a, embedding_p, embedding_n = embedding_b[0], embedding_b[1], embedding_b[2]
    h, w = embedding_b.shape[-2:]
    em_trans = lambda x: x.reshape((embedding_vec_dims, -1)).T
    summary_writer.add_image('BFVOS_Anchor/Embedding', projection_for_visualization(em_trans(embedding_a), 3, (h, w)), idx)
    summary_writer.add_image('BFVOS_Pos/Embedding', projection_for_visualization(em_trans(embedding_p), 3, (h, w)), idx)
    summary_writer.add_image('BFVOS_Neg/Embedding', projection_for_visualization(em_trans(embedding_n), 3, (h, w)), idx)

    im_trans = lambda x: (x.transpose((1, 2, 0)).astype('float')+np.array([122.675, 116.669, 104.008])).astype('uint8')
    summary_writer.add_image('BFVOS_Anchor/Image', im_trans(image_a), idx)
    summary_writer.add_image('BFVOS_Pos/Image', im_trans(image_p), idx)
    summary_writer.add_image('BFVOS_Neg/Image', im_trans(image_n), idx)


class LRScher(object):
    def __init__(self, base_lr, max_iter, power=0.9, logger=None):
        self.base_lr = base_lr
        self.max_iter = max_iter
        self.power = power
        self.logger = logger

    def adjust_lr(self, optimizer, cur_iter):
        if cur_iter <= 100:
            self.logger.info("Param group 0 LearningRate reaches {:03f}...".format(optimizer.param_groups[0]['lr']))
            self.logger.info("Param group 1 LearningRate reaches {:03f}...".format(optimizer.param_groups[1]['lr']))
        else:
            optimizer.param_groups[0]['lr'] = 2.5e-6 + (cur_iter/self.max_iter) * (2.5e-4-2.5e-6)
            optimizer.param_groups[1]['lr'] = 2.5e-6 + (cur_iter/self.max_iter) * (2.5e-4-2.5e-6)
            self.logger.info("Param group 0 LearningRate reaches {:03f}...".format(optimizer.param_groups[0]['lr']))
            self.logger.info("Param group 1 LearningRate reaches {:03f}...".format(optimizer.param_groups[1]['lr']))
            
            
        #cur_lr = self.base_lr*((1-float(cur_iter)/self.max_iter)**(self.power))
        #optimizer.param_groups[0]['lr'] = cur_lr
        #optimizer.param_groups[1]['lr'] = cur_lr * 2
        #if cur_iter % 10 == 0:
        #    self.logger.info("Param group 0 LearningRate reaches {:03f}...".format(optimizer.param_groups[0]['lr']))
        #    self.logger.info("Param group 1 LearningRate reaches {:03f}...".format(optimizer.param_groups[1]['lr']))

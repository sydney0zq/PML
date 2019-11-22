#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <qiang.zhou@yz-gpu029.hogpu.cc>
#
# Distributed under terms of the MIT license.

"""

"""

from dataset.davis import DAVISDataset
from net.bfnet import BFVOSNet
import numpy as np
import torch
import argparse
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import os
import cv2
from net.loss import minTripletLoss
import torch.nn as nn
from net.utils import projection_for_visualization
from net.bs import BSWrapper
from multiprocessing.dummy import Pool as ThreadPool
import time
from logger.logger import setup_logger

logger = setup_logger()

image_dims = [854, 480]
reduced_image_dims = [image_dims[1]//8+1, image_dims[0]//8+1]
embedding_vector_dims = 128
K=5
PARALLEL_BS=10
global loss_func
global bs_wrapper
mean_value = np.array([122.675, 116.669, 104.008]) 

def interp(x, out_size=(854, 480)):
    im = Image.fromarray(x)
    return np.asarray(im.resize(out_size, resample=Image.BILINEAR))

def dm(Pixels1, Pixels2):
    # Pixels: feat_dim * num_pixels

    sqPixels1 = np.reshape(np.sum(np.square(Pixels1), axis=0), (-1,1))
    sqPixels2 = np.reshape(np.sum(np.square(Pixels2), axis=0), (-1,1))

    sqDist = sqPixels1 + sqPixels2.T - 2 * np.dot(Pixels1.T, Pixels2)
    sqDist = np.maximum(sqDist, 0)
    return sqDist


class Postor:
    def __init__(self, ref_image_embedding, ref_mask, output_dir):
        self.ref_image_embedding = ref_image_embedding
        self.ref_mask = ref_mask
        self.output_dir = output_dir

    def process(self, cur_image_embedding, cur_image, i):
        ref_image_embedding = self.ref_image_embedding
        ref_mask = self.ref_mask
        output_dir = self.output_dir
        distances = dm(cur_image_embedding, ref_image_embedding)
        indices = np.argsort(distances, axis=1)[:, :K]
        output_mask = np.zeros(cur_image_embedding.shape[1]).flatten()
        output_mask[np.sum(ref_mask[indices], axis=1) > K/2] = 1
        output_mask = output_mask.reshape(reduced_image_dims)
        cv2.imwrite(os.path.join(output_dir, "{:05d}_raw.png".format(i)), output_mask * 255)
        output_mask_upsample = cv2.resize(output_mask, tuple(image_dims), interpolation=cv2.INTER_LINEAR)
        
        output_mask_final = bs_wrapper.solve(cur_image, output_mask_upsample)
        cv2.imwrite(os.path.join(output_dir, "{:05d}.png".format(i)), output_mask_final * 255)

def retrieve(dataobj, vdidx, model, output_dir):
    # Get reference assets
    ref_data = dataobj[vdidx[0]]
    ref_image_embedding = model(ref_data['image'].cuda().unsqueeze(0), ref_data['spatio_temporal_frame'].cuda().unsqueeze(0))
    ref_image_embedding = ref_image_embedding.cpu().numpy().reshape((embedding_vector_dims, -1))
    ref_mask = ref_data['annotation'].numpy().flatten()
    postor = Postor(ref_image_embedding, ref_mask, output_dir)

    # Batch inference
    pointer = 1
    pool = ThreadPool(processes=PARALLEL_BS)
    
    while pointer <= len(vdidx)-1:
        s = time.time()
        infer_bs = min(PARALLEL_BS, len(vdidx[pointer:pointer+PARALLEL_BS]))
        param_pool = []
        for fid, i in enumerate(vdidx[pointer:pointer+infer_bs]):
            cur_data = dataobj[i]
            cur_image = cur_data['image'].cuda().unsqueeze(0)
            spatio_temporal_frame = cur_data['spatio_temporal_frame'].cuda().unsqueeze(0)
            cur_image_embedding = model(cur_image, spatio_temporal_frame)
            cur_image_embedding = cur_image_embedding.data.cpu().numpy().reshape((embedding_vector_dims, -1))
            param_pool.append([cur_image_embedding, cur_data['image'].numpy().transpose((1, 2, 0))+mean_value, pointer+fid])
        pool.map(lambda x: postor.process(x[0], x[1], x[2]), param_pool)
        logger.info ("Processing frame {} to {}, costing {:03f}".format(pointer, pointer+infer_bs, time.time()-s))
        pointer += infer_bs
    pool.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help="Path to DAVIS directory")
    parser.add_argument('--seq_name', type=str, required=True,
                        help='Path to directory containing input image sequence')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained model weight')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory to save segmentation masks')
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--bsthres', type=float, default=0.3)
    args = parser.parse_args()

    global loss_func
    global bs_wrapper
    loss_func = minTripletLoss(alpha=args.alpha)
    bs_wrapper = BSWrapper(imsize=image_dims, thres=args.bsthres)

    dataobj = DAVISDataset(args.base_dir, image_dims, 2016, phase='val', split='val')
    model = BFVOSNet(embedding_vector_dims=embedding_vector_dims)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.freeze_bn()

    def retrieve_one_seq(dataobj, seq_name, model, output_dir):
        vdidx =  dataobj.sequence_to_sample_idx[seq_name]
        output_dir = os.path.join(output_dir, seq_name)
        os.makedirs(output_dir, exist_ok=True)    
        with torch.no_grad():
            retrieve(dataobj, vdidx, model, output_dir)

    logger.info ("Start to do evaluation...")
    if args.seq_name != "all":
        logger.info ("Now seq {}".format(args.seq_name))
        retrieve_one_seq(dataobj, args.seq_name, model, args.output_dir)
    else:
        for i_seq_name in dataobj.sequences:
            logger.info ("Now seq {}".format(i_seq_name))
            retrieve_one_seq(dataobj, i_seq_name, model, args.output_dir)

if __name__ == "__main__":
    main()


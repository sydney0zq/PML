#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <theodoruszq@gmail.com>


import argparse
import os
import sys
import numpy as np
import time
import datetime
import json

import logging
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset.davis import DAVISDataset, TripletSampler
from net import bfnet, loss
from tensorboardX import SummaryWriter
from logger.logger import setup_logger
from net.utils import visual_embeddings, create_triplet_pools, LRScher
from retrieve import retrieve
from davis_eval import compute_mIOU
import warnings
warnings.filterwarnings('ignore')

# Set paths
root_dir = "experiments"
data_dir = os.path.join("DAVIS")
model_dir = os.path.join(root_dir, "model")
training_dir = os.path.join(root_dir, "training")
deeplab_resnet_pre_trained_path = os.path.join("init_models", 'deeplabv2_voc.pth')
checkpoints_dir = os.path.join(training_dir, 'checkpoints')
config_save_dir = os.path.join(training_dir, 'configs')
tensorboard_save_dir = os.path.join(training_dir, 'tensorboard_logs')
output_dir = "test_output"

os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(config_save_dir, exist_ok=True)
os.makedirs(tensorboard_save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('-i', '--image_size', nargs=2, type=int, help="Input image dimension as [w, h]", metavar=('width', 'height'), default=[321, 321])
    parser.add_argument('-e', '--embedding_vector_dims', type=int, default=128, help="Embedding vector dims")
    # Intervals
    parser.add_argument('-c', '--checkpoint_interval', type=int, default=2, help="Epoch interval of saving models")
    parser.add_argument('--val_interval', type=int, default=2, help='Iterations after which to evaluate validation set')
    # Train
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Number of triplets in each batch')
    parser.add_argument('-g', '--gpus', type=str, default="0,1,2,3", help='GPUs to use')
    parser.add_argument('-n', '--num_epochs', type=int, default=50)
    parser.add_argument('-r', '--learning_rate', type=float, default=2.5e-6)
    # num_anchor_sample_points = 256  # according to paper
    parser.add_argument('-a', '--alpha', type=float, default=0.7, help='Slack variable for loss')
    parser.add_argument('-f', '--log_file', type=str, default=None,
                                    help='path to log file, setting this will log all messages to this file')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                                    help='Path to checkpoint file to resume training, otherwise train from scratch')
    parser.add_argument('--start_epoch', type=int, default=0,
                                    help='Epoch number to start')
    parser.add_argument('--lr_power', type=float, default=0.9)
    return parser.parse_args()

class Sphinx:
    def __init__(self):
        self.implant_param()
        self.init()

    def init(self):
        self.summary_writer = SummaryWriter(self.tensorboard_save_dir)
        self.logger = setup_logger() if self.log_file is None else setup_logger(self.log_file)
        self.train_loader, self.val_loader = self.get_dataloader()
        self.model, self.train_loss_fn, self.optimizer = self.get_model_loss_optimizer()
        self.lrscher = LRScher(self.learning_rate, self.num_epochs*len(self.train_loader), self.lr_power, self.logger)

    def implant_param(self):
        args = parse_args()
        self.args = args
        self.image_size = args.image_size
        self.embedding_vector_dims = args.embedding_vector_dims
        self.batch_size = args.batch_size
        self.alpha = args.alpha
        self.start_epoch = args.start_epoch
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.checkpoint_interval = args.checkpoint_interval
        self.lr_power = args.lr_power
        # OS
        self.checkpoint_path = args.checkpoint_path
        self.tensorboard_save_dir = tensorboard_save_dir
        self.deeplab_resnet_pre_trained_path = deeplab_resnet_pre_trained_path
        self.log_file = args.log_file
        self.model_dir = model_dir
        self.config_save_dir = config_save_dir
        self.checkpoints_dir = checkpoints_dir
        self.output_dir = output_dir
        self.devices = [int(x) for x in args.gpus.split(',')]

    def get_dataloader(self):
        train_data_source = DAVISDataset(baseDir=data_dir, image_size=self.image_size, year=2016, phase='train', split='train')
        train_triplet_sampler = TripletSampler(dataset=train_data_source, num_triplets=self.batch_size, randomize=True)
        train_data_loader = DataLoader(dataset=train_data_source, batch_sampler=train_triplet_sampler, num_workers=(self.batch_size+1)//2)
        val_data_source = DAVISDataset(baseDir=data_dir, image_size=self.image_size, year=2016, phase='train', split='val')
        val_triplet_sampler = TripletSampler(dataset=val_data_source, num_triplets=self.batch_size, randomize=True)
        val_data_loader = DataLoader(dataset=val_data_source, batch_sampler=val_triplet_sampler, num_workers=(self.batch_size+1)//2)
        
        return [train_data_loader, val_data_loader]

    def get_model_loss_optimizer(self):
        model = bfnet.BFVOSNet(embedding_vector_dims=self.embedding_vector_dims, logger=self.logger)
        train_loss_fn = loss.minTripletLoss(alpha=self.alpha)

        if self.checkpoint_path is not None:
            model.load_state_dict(torch.load(self.checkpoint_path))
            self.logger.info("Loaded checkpoint from {}...".format(self.checkpoint_path))
        else:
            ori_state_dict = torch.load(self.deeplab_resnet_pre_trained_path)
            cur_state_dict = model.state_dict()
            for k in ori_state_dict.keys():
                cur_k = "base_extractor." + k
                if cur_k in cur_state_dict.keys():
                    cur_state_dict[cur_k] = ori_state_dict[k]
            model.load_state_dict(cur_state_dict)
            self.logger.info("Loaded DeepLab ResNet from {}...".format(self.deeplab_resnet_pre_trained_path))
        model.train()
        #model.freeze_feature_extraction()   # NOTE
        #model.freeze_bn(name="base_extractor")

        #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
        #                          lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.SGD([{'params': model.get_1x_lr_params(),  'lr': self.learning_rate }, 
                               {'params': model.get_10x_lr_params(), 'lr': 2*self.learning_rate}], 
                               lr=self.learning_rate, momentum=0.9, weight_decay=5e-3)
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
        #                          lr=self.learning_rate)
        model = nn.DataParallel(model, device_ids=self.devices)
        return [model.cuda(), train_loss_fn.cuda(), optimizer]

    def train(self):
        data_len = len(self.train_loader)
        self.model.module.check_bn('base_extractor')
        self.model.module.check_bn('embedding_head')
        for epoch in range(self.start_epoch, self.num_epochs):
            self.cur_epoch = epoch
            for i_loader, sample in enumerate(self.train_loader):
                embeddings = self.model(sample['image'].cuda(), sample['spatio_temporal_frame'].cuda())
                fg_loss, bg_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
                # sample_frames and embeddings are triplets concatenated together. Split them out into triplet frames
                bs = int(sample['image'].size(0) / 3)
                for i_bs in range(bs):
                    triplet_sample = {}
                    for key in sample:
                        triplet_sample[key] = sample[key][3*i_bs : 3*i_bs+3]
                    triplet_embeddings = embeddings[3*i_bs : 3*i_bs+3]
                    triplet_pools = create_triplet_pools(triplet_sample, triplet_embeddings, phase='train')

                    if triplet_pools is None:
                        # Skip as not enough triplet samples were generated (possibly due to downsampled/low-res ground truth)
                        self.logger.warning("Skipping epoch {}, i_loader {}, i_bs {}".format(epoch, i_loader, i_bs))
                        continue
                    else:
                        fg_embedding_a, bg_embedding_a, fg_positive_pool, fg_negative_pool, bg_positive_pool, bg_negative_pool = triplet_pools
                    fg_loss += self.train_loss_fn(fg_embedding_a, fg_positive_pool, fg_negative_pool)
                    bg_loss += self.train_loss_fn(bg_embedding_a, bg_positive_pool, bg_negative_pool)

                fg_loss /= bs
                bg_loss /= bs
                final_loss = (fg_loss + bg_loss) * 0.5

                abs_i_loader = epoch*data_len+i_loader
                self.logger.info("TRAIN(E{:02d}/I{:04d}): fg_loss={:05f}, bg_loss={:05f}, final_loss={:05f}".format(\
                                                            epoch, abs_i_loader, fg_loss.item(), bg_loss.item(), final_loss.item()))
                self.summary_writer.add_scalar('BFVOS_TRAIN/fg_loss', fg_loss.item(), abs_i_loader)
                self.summary_writer.add_scalar('BFVOS_TRAIN/bg_loss', bg_loss.item(), abs_i_loader)
                self.summary_writer.add_scalar('BFVOS_TRAIN/final_loss', final_loss.item(), abs_i_loader)

                # Visualize embedding
                if abs_i_loader % 10 == 0:
                    visual_embeddings(embeddings, sample['image'], self.summary_writer, abs_i_loader)
                # Backpropagation
                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()
                self.lrscher.adjust_lr(self.optimizer, abs_i_loader)

            if epoch % self.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.checkpoints_dir, "ckpt_epoch_{:02d}.pth".format(epoch))
                torch.save(self.model.state_dict(), ckpt_path)
                self.logger.info("Checkpoint saved at {}...".format(ckpt_path))

            self.validate()
            if epoch % 5 == 0:
                self.test()
            self.logger.info("Finished epoch {} with {} iterations...".format(epoch, i_loader))
        self.test()
            
        # Save final model after all epochs
        save_model_filename = "epoch_{:02d}_{}.model".format(self.num_epochs,
                str(datetime.datetime.now()).replace(" ", "_").replace(":", "_"))
        save_model_path = os.path.join(self.model_dir, save_model_filename)
        torch.save(self.model.state_dict(), save_model_path)
        self.logger.info("Model saved to {}...".format(save_model_path))
        training_config_save_path = os.path.join(self.config_save_dir, save_model_filename.replace('.model', '.json'))
        training_config = vars(self.args)
        with open(training_config_save_path, 'w') as f:
            json.dump(training_config, f, indent=1)
        self.logger.info("Training config saved to {}...".format(training_config_save_path))

    def validate(self):
        val_fg_loss, val_bg_loss, val_final_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        self.logger.warning("Start to evaluate...")
        self.model.eval()
        self.model.module.check_bn('embedding_head')
        with torch.no_grad():
            for idx, sample in enumerate(self.train_loader):
                if idx >= 100:
                    break
                fg_loss, bg_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
                embeddings = self.model(sample['image'].cuda(), sample['spatio_temporal_frame'].cuda())
                bs = int(sample['image'].size(0) / 3)
                for i_bs in range(bs):
                    triplet_sample = {}
                    for key in sample:
                        triplet_sample[key] = sample[key][3*i_bs : 3*i_bs+3]
                    triplet_embeddings = embeddings[3*i_bs : 3*i_bs+3]
                    triplet_pools = create_triplet_pools(triplet_sample, triplet_embeddings, phase='val')

                    if triplet_pools is None:
                        # Skip as not enough triplet samples were generated (possibly due to downsampled/low-res ground truth)
                        self.logger.warning("Skipping epoch {}, i_loader {}, i_bs {}".format(self.cur_epoch, idx, i_bs))
                        continue
                    else:
                        fg_embedding_a, bg_embedding_a, fg_positive_pool, fg_negative_pool, bg_positive_pool, bg_negative_pool = triplet_pools
                    fg_loss += self.train_loss_fn(fg_embedding_a, fg_positive_pool, fg_negative_pool)
                    bg_loss += self.train_loss_fn(bg_embedding_a, bg_positive_pool, bg_negative_pool)
                fg_loss /= bs
                bg_loss /= bs
                final_loss = (fg_loss + bg_loss) * 0.5
                if idx % 10 == 0:
                    self.logger.info("VAL(E{:02d}/I{:04d}): fg_loss={:05f}, bg_loss={:05f}, final_loss={:05f}".format(\
                                                                self.cur_epoch, idx, fg_loss.item(), bg_loss.item(), final_loss.item()))
                val_fg_loss += fg_loss
                val_bg_loss += bg_loss
                val_final_loss += final_loss
            val_fg_loss = val_fg_loss / (idx+1)
            val_bg_loss = val_bg_loss / (idx+1)
            val_final_loss = val_final_loss / (idx+1)
            self.logger.info("VAL(E{:02d}): fg_loss={:05f}, bg_loss={:05f}, final_loss={:05f}".format(\
                                                        self.cur_epoch, val_fg_loss.item(), val_bg_loss.item(), val_final_loss.item()))
            self.summary_writer.add_scalar('BFVOS_VAL/fg_loss', val_fg_loss.item(), self.cur_epoch)
            self.summary_writer.add_scalar('BFVOS_VAL/bg_loss', val_bg_loss.item(), self.cur_epoch)
            self.summary_writer.add_scalar('BFVOS_VAL/final_loss', val_final_loss.item(), self.cur_epoch)
        self.model.train()
        #self.model.module.freeze_bn(name="base_extractor")
        self.model.module.check_bn('base_extractor')
        self.model.module.check_bn('embedding_head')

    def test(self):
        self.model.eval()
        self.model.module.check_bn('embedding_head')
        self.logger.info("Start to do test...")
        image_dims = [854, 480]
        reduced_image_dims = [image_dims[1]//8+1, image_dims[0]//8+1]
        dataobj = DAVISDataset(data_dir, image_dims, 2016, phase='val', split='val')
        output_dir = os.path.join(self.output_dir, "epoch_{}".format(self.cur_epoch))
        def retrieve_one_seq(dataobj, seq_name, model, output_dir):
            vdidx =  dataobj.sequence_to_sample_idx[seq_name]
            output_dir = os.path.join(output_dir, seq_name)
            os.makedirs(output_dir, exist_ok=True)
            with torch.no_grad():
                retrieve(dataobj, vdidx, model, output_dir, logger=self.logger)
        for i_seq_name in dataobj.sequences:
            self.logger.info ("Now seq {}".format(i_seq_name))
            retrieve_one_seq(dataobj, i_seq_name, self.model, output_dir)
        iou = compute_mIOU(os.path.join(data_dir, "trainval"), output_dir, 2016, 'val', logger=self.logger)
        self.summary_writer.add_scalar('BFVOS_TEST/mIOU', iou, self.cur_epoch)
        self.model.train()
        #self.model.module.freeze_bn(name="base_extractor")
        self.model.module.check_bn('base_extractor')
        self.model.module.check_bn('embedding_head')


if __name__ == "__main__":
    Trainer = Sphinx()
    Trainer.train()





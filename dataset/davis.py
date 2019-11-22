#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <theodoruszq@gmail.com>

from torch.utils.data import Dataset, sampler
import os
from natsort import natsorted
import numpy as np
from PIL import Image
import torch
import cv2
import random
import sys
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR)
from datautils import get_mask_bbox, cropimage, Auger

""" Convert numpy array to PyTensors """
class ToTensor(object):
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': torch.from_numpy(sample['image'].transpose((2, 0, 1))).type(torch.FloatTensor),
                'annotation': torch.from_numpy(sample['annotation'].astype(np.uint8)),
                'spatio_temporal_frame': torch.from_numpy(sample['spatio_temporal_frame'].transpose((2, 0, 1))).type(torch.FloatTensor),
                'frame_no': sample['frame_no'],
                'seq_name': sample['seq_name']
                }


class DAVISDataset(Dataset):
    def __init__(self, baseDir, image_size, year, phase='train', split='unittest', transform=ToTensor()):
        """
            :baseDir: DAVIS dataset root
            :image_size: (w, h) tuple to resize the image
            :year: 2016/2017
            :phase: train/val/test
            :transform: ToTensor function
        """
        super().__init__()
        self._baseDir = baseDir
        self._image_size = image_size
        self._reduced_image_size = [self._image_size[0]//8+1, self._image_size[1]//8+1]
        self.mean_value = np.array([122.675, 116.669, 104.008])
        self.auger = Auger(cropsize=image_size, mirror_aug=True)
        self.phase = phase
        if phase in ['train', 'val', 'trainval']:
            self._baseDir = os.path.join(self._baseDir, 'trainval')
        elif phase in ['test', 'testdev']:
            self._baseDir = os.path.join(self._baseDir, 'testdev')
        elif phase == 'testchallenge':
            self._baseDir = os.path.join(self._baseDir, 'testchallenge')
        self._images_dir = os.path.join(self._baseDir, 'JPEGImages/480p')
        self._annotations_dir = os.path.join(self._baseDir,  'Annotations/480p')
        self._transform = transform
        if year is not None:
            assert(year==2016 or year==2017)
            years = [year]
        else:
            years = [2016, 2017]
        self.sequences = []
        for year in years:
            with open(os.path.join(self._baseDir, 'ImageSets/{}/{}.txt'.format(year, split)), 'r') as f:
                self.sequences += [seq.strip() for seq in f.readlines()]
        self.sequences = sorted(list(set(self.sequences)))

        # Store all image paths, annotations paths, frame numbers and sequence labels
        self._frame_data = []
        for seq in self.sequences:
            # Each frame is a tuple contains (image_path, anno_path, frame_id, video_name)
            self._frame_data += list(map(lambda x: (
                os.path.join(self._images_dir, seq, x),
                os.path.join(self._annotations_dir, seq, x.replace('.jpg', '.png')),
                int(os.path.splitext(x)[0]), seq), natsorted(os.listdir(os.path.join(self._images_dir, seq)))))

        # Easy access to sequence specific framedata
        self.sequence_to_sample_idx = {seq: [] for seq in self.sequences}
        for idx, sample in enumerate(self._frame_data):
            self.sequence_to_sample_idx[sample[3]].append(idx)

    def __len__(self):
        return len(self._frame_data)

    def __getitem__(self, idx):
        # Load the <idx>th image and annotation and return a sample
        image_path, annotation_path, frame_no, seq_name = self._frame_data[idx]
        image = Image.open(image_path).convert('RGB')
        annotation = Image.open(annotation_path).convert('L')
        
        if self.phase == "train":
            # w/ Augmentation
            #image, annotation = self.auger.aug(image, annotation)
            #image = Image.fromarray(image.astype(np.uint8))
            #annotation = Image.fromarray(annotation.astype(np.uint8))

            if image.size != self._image_size: image = image.resize(self._image_size)
            annotation = annotation.resize(self._reduced_image_size, Image.NEAREST)
        elif self.phase == "val":
            if image.size != self._image_size:
                image = image.resize(self._image_size)
            annotation = annotation.resize(self._reduced_image_size, Image.NEAREST)
        
        image = np.array(image).astype(np.float) - self.mean_value   # Always RGB format

        # Spatial frame is array of hxwx2 where spatial_frame[h][w] = [h, w]
        spatial_frame = np.dstack(np.mgrid[:self._reduced_image_size[1], :self._reduced_image_size[0]])
        spatial_frame[:, :, 0] = spatial_frame[:, :, 0] / self._reduced_image_size[1]
        spatial_frame[:, :, 1] = spatial_frame[:, :, 1] / self._reduced_image_size[0]
        #temporal_frame = np.ones(self._reduced_image_size) * frame_no
        #spatio_temporal_frame = np.dstack((spatial_frame, temporal_frame))
        spatio_temporal_frame = np.asarray(spatial_frame, dtype='float')
        
        # Convert annotation to binary image
        annotation = np.asarray(annotation).copy()
        annotation[annotation > 0] = 1
        sample = {
                'image': image,
                'annotation': annotation,
                'spatio_temporal_frame': spatio_temporal_frame,
                'frame_no': frame_no,
                'seq_name': seq_name
        }
        if self._transform:
            sample = self._transform(sample)
        return sample


class TripletSampler(sampler.Sampler):
    def __init__(self, dataset, sequence=None, randomize=True, num_triplets=1):
        super().__init__(data_source=dataset)
        self._dataset = dataset
        self._randomize = randomize
        self._num_triplets = num_triplets
        if sequence is not None:
            self._num_samples = len(self._dataset.sequence_to_sample_idx[sequence]) // num_triplets
            self._sequences = [sequence]
        else:
            self._num_samples = len(self._dataset) // num_triplets
            self._sequences = list(self._dataset.sequence_to_sample_idx.keys())

    def __iter__(self):
        for i in range(self._num_samples):
            triplets = []
            for i_trip in range(self._num_triplets):
                seq = random.choice(self._sequences)
                seq_sample_idx = self._dataset.sequence_to_sample_idx[seq]
                a, p, n = np.random.choice(seq_sample_idx, size=3, replace=False)
                triplets += [a, p, n]
            yield triplets

        #for seq in self._sequences:
        #    print ("Current sequence: {}".format(seq))
        #    triplets = []
        #    # Now create set of random (a, p, n) samples from this sequence
        #    seq_sample_idx = self._dataset.sequence_to_sample_idx[seq]
        #    #for i in range(len(seq_sample_idx) * len(seq_sample_idx)):
        #        # Randomly sample two non-anchor separate frames - one for positive pool and one for negative
        #    for i in range(len(self._dataset.sequence_to_sample_idx[seq])//self._num_triplets):
        #        for i_trip in range(self._num_triplets):
        #            a, p, n = np.random.choice(seq_sample_idx, size=3, replace=False)
        #            triplets += [a, p, n]

        #        if len(triplets) % (3 * self._num_triplets) == 0:
        #            yield triplets
        #            triplets = []

    def __len__(self):
        return self._num_samples

if __name__ == "__main__":
    dd = DAVISDataset("/home/users/qiang.zhou/workspace/media/DAVIS", [321, 321], 2016, split='train')
    train_triplet_sampler = TripletSampler(dataset=dd, num_triplets=16, randomize=True) 
    train_data_loader = torch.utils.data.DataLoader(dataset=dd, batch_sampler=train_triplet_sampler, num_workers=16)
    for epoch in range(20):
        for idx, _ in enumerate(train_data_loader):
            print ("load epoch {} idx {}".format(epoch, idx))

    import pdb
    pdb.set_trace()

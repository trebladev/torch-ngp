import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, rotation

import trimesh

import torch
from torch.utils.data import dataloader

class NeRFDataset:
    def __init__(self, opt, device, type='train', n_test=10):
        super().__init__()

        self.opt = opt
        self.device = device
        # self.type = type # train, test, val
        # self.downscale = opt.downscale
        self.root_path = opt.path    
        # self.preload = opt.preload # preload data into gpu
        # self.scale = opt.scale
        # self.offset = opt.offset
        # self.bound = opt.bound
        # self.fp16 = opt.fp16
        
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap'
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender'
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms.json or transforms*.json under {self.root_path}')
        
        # load nerf-compatible format data
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
                        
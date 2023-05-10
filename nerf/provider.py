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

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    pose[:3, 3] = pose[:3, 3] * scale + np.array(offset)
    pose = pose.astype(np.float32)
    return pose

def visualize_pose(poses, size=0.1, bound=1):
    # pose: [B, 4, 4]
    
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if bound > 1:
        unit_box = trimesh.primitives.Box(extents=[2]*3).as_outline()
        unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
        objects.append(unit_box)

    for pose in poses:
        # a camera is visualized with 8 line segments
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c +d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


class NeRFDataset:
    def __init__(self, opt, device, type='train', n_test=10):
        super().__init__()

        self.opt = opt
        self.device = device
        # self.type = type # train, test, val
        # self.downscale = opt.downscale
        self.root_path = opt.path    
        # self.preload = opt.preload # preload data into gpu
        self.scale = opt.scale
        self.offset = opt.offset
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
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    tmp_transform = json.load(f)
                transform['frames'].extend(tmp_transform['frames'])
            else:
                with open(os.path.join(self.root_path, f"transform_{type}.json"), 'r') as f:
                    transform = json.load(f)
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'])
            self.W = int(transform['w'])
        else:
            # read image size later
            self.H = self.W = None

        # read images
        frames = np.array(transform['frames'])

        # for colmap, manually interpolate a test set
        if self.mode == 'colmap' and type == 'test':

            # TODO the first version use blender datasets, so colmap will later to implement
            # choose two random poses, and interpolate between
            f0, f1 = np.random.choice(frames, 2, replace=False)

        else:
            # for colmap, manually split a valid set (the first frame)
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval': use all frames

            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png'
                
                # there are non exists paths in fox
                if not os.path.exists(f_path):
                    print(f'Warning: {f_path} does not exist')
                    continue
                
                # get poses
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] or [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0]  # self.downscale
                    self.W = image.shape[1]  # self.downscale

                # add support for the alpha channel as a mask
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                
                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H,), interpolation=cv2.INTER_AREA)

                self.poses.append(pose)
                self.images.append(image)

        # convert image to tensor
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4] 
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0).astype(np.uint8)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        # if self.opt.vis_pose:
            # visualize_pose()
        visualize_pose(self.poses.numpy(), bound=1)




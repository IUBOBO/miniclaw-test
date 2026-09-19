import logging
import torch
import numpy as np
from os.path import splitext
from os import listdir
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, imgs_dir, boundary_dir, mask_dir):
        self.imgs_dir = imgs_dir
        self.boundary_dir = boundary_dir
        self.mask_dir = mask_dir

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)  # 单通道图像增加通道维度

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans / 255.0  # 归一化到 [0, 1]
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        boundary_file = glob(os.path.join(self.boundary_dir, idx + '.*'))
        img_file = glob(os.path.join(self.imgs_dir, idx + '.*'))
        mask_file = glob(os.path.join(self.mask_dir, idx + '.*'))

        assert len(boundary_file) == 1, f'Either no boundary or multiple boundaries found for the ID {idx}: {boundary_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'

        boundary = Image.open(boundary_file[0])
        img = Image.open(img_file[0])
        mask = Image.open(mask_file[0])

        assert img.size == boundary.size, f'Image and boundary {idx} should be the same size, but are {img.size} and {boundary.size}'
        assert img.size == mask.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img)
        boundary = self.preprocess(boundary)
        mask = self.preprocess(mask)

        return {
            'image': torch.from_numpy(img).float(),  # 转换为浮点数张量
            'boundary': torch.from_numpy(boundary)[0][None].float(),  # 边界 (1, H, W)
            'mask': torch.from_numpy(mask)[0][None].float(),  # 掩码 (1, H, W)

        }



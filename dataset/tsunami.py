import os
from os.path import join
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class Dataset(Dataset):
    def __init__(self,dataset,transform_med):
        self.batch_size = dataset['batch_size']
        self.splits = dataset['splits']
        self.dataset_name = dataset['dataset_name']
        self.data_root = join(dataset['data_root'],self.dataset_name)
        self.data_txt_dir = join(self.data_root,dataset['dataset_txt_root'])
        self.data_augment = dataset['data_augment']
        self.mean_value_t0 = dataset['mean_value_t0']
        self.mean_value_t1 = dataset['mean_value_t1']
        self.img_gt_pairs = self.img_gt_pair()

    def img_gt_pair(self):

        img_gt_dict = dict()
        imgs = np.loadtxt(self.data_txt_dir,dtype=str)
        for idx, (img1_dir,img2_dir,gt_dir) in enumerate(imgs):
            name = gt_dir.split('/')[1]
            img1 = join(self.data_root,img1_dir)
            img2 = join(self.data_root,img2_dir)
            gt = join(self.data_root,gt_dir)
            img_gt_dict.setdefault(idx,[img1,img2,gt,name])
        return img_gt_dict

    def data_transform(self, img1,img2,lbl):
        img1 = img1[:, :, ::-1]  # RGB -> BGR
        img1 = img1.astype(np.float64)
        img1 -= self.mean_value_t0
        img1 = img1.transpose(2, 0, 1)
        img1 = torch.from_numpy(img1).float()
        img2 = img2[:, :, ::-1]  # RGB -> BGR
        img2 = img2.astype(np.float64)
        img2 -= self.mean_value_t1
        img2 = img2.transpose(2, 0, 1)
        img2 = torch.from_numpy(img2).float()
        lbl = torch.from_numpy(lbl).long()
        return img1,img2,lbl

    def __getitem__(self, item):

        img1_dir, img2_dir, gt_dir, name = self.img_gt_pairs[item]
        # Extract image as PyTorch tensor
        img1,img2 = Image.open(img1_dir),Image.open(img2_dir)
        height,width,_ = np.array(img1,dtype= np.uint8).shape
        img1 = np.array(img1,dtype= np.uint8)
        img2 = np.array(img2,dtype= np.uint8)
        label = Image.open(gt_dir)
        label = np.array(label, dtype=np.int32)
        img1, img2, label = self.data_transform(img1, img2, label)
        return img1, img2, label, name

    def __len__(self):
        return len(self.img_gt_pairs)

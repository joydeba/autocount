import torch
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import numpy as np
from xml.dom import minidom 
import cv2
import math 

class DataLoaderInstanceSegmentation(Dataset):
    def __init__(self, folder_path="imagesSample", train = True):
        super(DataLoaderInstanceSegmentation, self).__init__()
        if train:
            folder_path="imagesSample"
        else:     
            folder_path="images_testing"
        self.train = train
        self.img_files = glob.glob(os.path.join(folder_path,"raw","*.jpg"))
        self.ins_mask_files = []
        self.filenames = []
        self.to_tensor = transforms.ToTensor()
        for img_path in self.img_files:
            self.ins_mask_files.append(os.path.join(folder_path,'croped_masks',os.path.basename(img_path)))
            self.filenames.append(os.path.basename(img_path))


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        ins_mask_path = self.ins_mask_files[index]
        filename = self.filenames[index]

        # data =  np.asarray(Image.open(img_path).convert('RGB')).transpose((2,0,1))
        # data = torch.Tensor(data)
        data =  self.to_tensor(Image.open(img_path).convert('RGB'))

        # label_ins =  np.asarray(Image.open(ins_mask_path).convert('RGB')).transpose((2,0,1))
        # label_ins = torch.Tensor(label_ins)
        label_ins =  self.to_tensor(Image.open(ins_mask_path).convert('RGB'))



        if self.train:
            return data, label_ins
        else:     
            return data, filename
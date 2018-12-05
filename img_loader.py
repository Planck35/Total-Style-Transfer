import os
import torch
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import scipy.misc
from torch.utils.serialization import load_lua
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# from model import Encoder, Decoder, StyleLoss, ContentLoss


class MyCostumeDataset(Dataset):
    def __init__(self, style, content):
        '''
        style:      n*3*W*H imgs - [PIL]
        content:    m*3*W*H imgs - [PIL]
        '''
        self.train_s = {}
        self.train_c = {}

        self.data_len = len(style) * len(content)
        self.idx = [(i, j) for i in range(len(style)) for j in range(len(content))]

        self.train_prep = transforms.Compose([
                    # transforms.Resize(size=(256, 256)),
                    transforms.RandomCrop(240),
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    ])
        for i, img in enumerate(style):
            ratio = 256/min(img.size)
            resize = (256, int(img.size[1] * ratio)) if img.size[0] < img.size[1] else (int(img.size[1] * ratio), 256)
            img = transforms.functional.resize(img, resize)
            self.train_s[i] = img
        for i, img in enumerate(content):
            ratio = 256/min(img.size)
            resize = (256, int(img.size[1] * ratio)) if img.size[0] < img.size[1] else (int(img.size[1] * ratio), 256)
            img = transforms.functional.resize(img, max(img.size))
            self.train_c[i] = img

    def __getitem__(self, index):
        (sid, cid) = self.idx[index]
        style = self.train_s[sid]
        content = self.train_c[cid]
        style = self.train_prep(style)
        content = self.train_prep(content)
        style = torch.tensor(style)
        content = torch.tensor(content)

        if style.size()[0] != 3:
            style = style.repeat(3, 1, 1)
        if content.size()[0] != 3:
            content = content.repeat(3, 1, 1)
        return style,content

    def __len__(self):
        return len(self.idx)


def file_name(file_dir):
    #print (file_dir)
    for root, dirs, files in os.walk(file_dir):
        # print(root) 
        # print(dirs) 
        #print(files)
        return files


def get_data_loader(content_path, style_path, batch_size, small_test = False):
    cfiles = file_name(content_path)
    sfiles = file_name(style_path)
    content_imgs = [Image.open(content_path+name) for name in cfiles]
    style_imgs = [Image.open(style_path+name) for name in sfiles]
    
    if small_test:
        content_imgs = content_imgs[:50]
        style_imgs = style_imgs[:50]

    dataset = MyCostumeDataset(style=style_imgs, content=content_imgs)
    train_loader = DataLoader(dataset, batch_size=batch_size , shuffle=True)
    print ("----------------------Data is loaded----------------------------")
    print ("Training Dataset: ", len(dataset))
    return train_loader

# content_path = "./data/content_img/"
# style_path = "./data/style_img/"
# train_loader = get_data_loader(content_path, style_path, 32, small_test = True)
# for idx, (style_img, content_img) in enumerate(train_loader):
#     print(style_img.size())
#     print(content_img.size())
#     break

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tkinter.messagebox import NO
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# input RGB image path
def img_rgb_loader(path):
    img =  cv2.imread(path)
    if img.shape[0]>480:
        img = cv2.resize(img, (480, 270), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =  img[150:250, :, :]
    return img

def img_rgb_loader120(path):
    img =  cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # if img.shape[0] > 150:
    img =  img[10:110, :, :]
    # img =  img[130:200, 0:480, 0:3]
    return img

def img_5gray_loader(path):
    gray_img_lists = []
    for i in range(len(path)):
        img_gray = cv2.imread(path[i]) 
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        gray_img_lists.append(img_gray)
    gray_imgs = cv2.merge(gray_img_lists)
    return gray_imgs

class img_dataset_out2(Dataset):
    def __init__(self, txt, transform=None, loader=img_rgb_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]), int(words[2]), int(words[3])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_path, label1, label2, scene = self.imgs[index]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label1, label2, img_path, scene

    def __len__(self):
        return len(self.imgs)

def img_rgb_random_loader(path, rnd):
    step = int(rnd * 40)
    area_begin, area_end = 120, 220
    img =  cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    begin, end = area_begin + step, area_end + step
    img =  img[begin:end, :, 0:3]
    return img

def img_rgb_random_loader120(path, rnd):
    if rnd > 0.5:
        paths = path.split('/')
        path = path.replace(paths[6], paths.split('_')[1])
    step = int(rnd * 20)
    area_begin, area_end = 0, 100
    img =  cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    begin, end = area_begin + step, area_end + step
    img =  img[begin:end, :, 0:3]
    return img

# 做了"随机翻转"、"随机裁剪"的数据增强
class img_dataset_out2_v2(Dataset):  
    def __init__(self, txt, transform=None, loader=img_rgb_random_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]), int(words[2]), int(words[3])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_path, label1, label2, scene = self.imgs[index]
        rnd = random.random()           # 0-1之间抽样随机数
        img = self.loader(img_path, rnd)
        if self.transform is not None:
            img = self.transform(img)
        return img, label1, label2, img_path, scene

    def __len__(self):
        return len(self.imgs)


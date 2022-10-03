#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
import random

def img_rgb_loader120(path):
    img =  cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img =  img[170:340, :, :]
    return img

class img_dataset_out2(Dataset):
    def __init__(self, txt, transform=None, loader=img_rgb_loader120):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]), int(words[2]), int(words[2])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_path, label1, label2, scene = self.imgs[index]
        # img_id = os.path.split(fn)[-1]
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label1, label2, img_path, scene

    def __len__(self):
        return len(self.imgs)

def img_random_loader120(path, rnd):
    step = int(rnd * 20)
    area_begin, area_end = 150, 320
    img =  cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    begin, end = area_begin + step, area_end + step
    img =  img[begin:end, :, :]
    return img

# 做了"随机翻转"、"随机裁剪"的数据增强
class img_dataset_out2_v2(Dataset):  
    def __init__(self, txt, transform=None, loader=img_random_loader120):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1]), int(words[2]), int(words[2])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_path, label1, label2, scene= self.imgs[index]
        rnd = random.random()           # 0-1之间抽样随机数
        img = self.loader(img_path, rnd)
        if self.transform is not None:
            img = self.transform(img)
        return img, label1, label2, img_path, scene

    def __len__(self):
        return len(self.imgs)


class classifer_head(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(classifer_head, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 512, kernel_size=(6, 15), stride=1, padding=0, bias=False)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.conv1(x)    #torch.Size([batch, 512, 1, 1])
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ShareResNet_Out2(nn.Module):
    def __init__(self, input_ch, num_class, pretrained=True):
        super(ShareResNet_Out2, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        # 把最后的Avgpool和Fully Connected Layer去除
        self.backbone = nn.Sequential(*list(self.model.children())[:-3])
        self.head1 = nn.Sequential(*list(self.model.children())[-3],
                                    classifer_head(512, num_class))
        self.head2 = nn.Sequential(*list(self.model.children())[-3], 
                                    classifer_head(512, num_class))

    def forward(self, x):
        x1 = x[:, :, :, 0:480]       #  torch.Size([4, 3, 100, 240]
        x1 = torch.flip(x1, [3])     #  将x的第3维水平翻转
        x1 = self.backbone(x1)       #  out: torch.Size([4, 512, 4, 8]
        output1 = self.head1(x1)

        x2 = x[:, :, :, 480:960]     #  torch.Size([4, 3, 100, 240]
        x2 = self.backbone(x2)
        output2 = self.head2(x2)
        return output1, output2

class Classifer(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(Classifer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        
class ResNet18_Out2(nn.Module):
    def __init__(self, input_ch, num_class, pretrained=True):
        super(ResNet18_Out2, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        # 把最后的Avgpool和Fully Connected Layer去除
        self.backbone = nn.Sequential(*list(self.model.children())[:-3])
        self.classification_head1 = nn.Sequential(*list(self.model.children())[-3],
                                                  Classifer(512, num_class))
        self.classification_head2 = nn.Sequential(*list(self.model.children())[-3],
                                                  Classifer(512, num_class))

    def forward(self, x):
        x = self.backbone(x)
        output1 = self.classification_head1(x)
        output2 = self.classification_head2(x)
        return output1, output2

def load_pretrain(model):
    model_dict = model.state_dict()
    dict_file = './result/pth/ShareNet-infov60-out0-4-20220607Bep0.pth'
    pretrained_dict = torch.load(dict_file)
    for k, v in pretrained_dict.items():
        print('load layer', k)
        if not(k in model_dict and v.shape == model_dict[k].shape):
            print("Attention! defined model can't matches perfectly with pretrained params")
            break
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
    # print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

# """-------for debug-------"""
# from torch.utils.data import DataLoader
# from torchvision import transforms
# if __name__ == '__main__':
#     net = ShareResNet_Out2(3, 7)
#     # net = load_pretrain(net)
#     root = "/disk2/dianzheng/lane_loc/lane_data/"
#     train_data = img_dataset_out2(txt=root + 'txt_cnn/test_data_cu.txt', transform=transforms.ToTensor())
#     train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
#     for images, labels1, labels2, img_file in train_loader:
#         print("images shape", images.shape)
#         print('labels1', labels1)
#         print('img file', img_file[0])
#         outputs1, outputs2 = net(images)
#         print("output shape", outputs1.shape)
#         # exit()

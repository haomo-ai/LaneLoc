#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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

class classifer_head(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(classifer_head, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 512, kernel_size=(4, 8), stride=1, padding=0, bias=False)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.conv1(x)    #torch.Size([4, 512, 1, 1])
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

# """-------for debug-------"""
# from dataset import img_dataset_out2
# from torch.utils.data import DataLoader
# from torchvision import transforms
# if __name__ == '__main__':
#     # net = models.resnet18(pretrained=True)  #input_channel, output_nums
#     net = ShareResNet18_Out2(3, 6)
#     # print(net)
#     root = "/disk2/dianzheng/lane_loc/lane_data/"
#     train_data = img_dataset_out2(txt=root + 'test_case.txt', transform=transforms.ToTensor())
#     train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
#     for i, (images, labels1, labels2, img_file) in enumerate(train_loader):
#         # img_file = img_file.to(torch.device("cuda:0"))
#         print("images shape", images.shape)
#         print('labels1', labels1)
#         print('labels2', labels2)
#         print('img file', img_file[0])
#         outputs1, outputs2 = net(images)
#         print("output shape", outputs1.shape)
#         if(i == 3):
#             break

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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
        x1 = x[:, :, :, :240]       #  torch.Size([4, 3, 100, 240]
        x1 = torch.flip(x1, [3])     #  将x的第3维水平翻转
        x1 = self.backbone(x1)       #  out: torch.Size([4, 512, 4, 8]
        output1 = self.head1(x1)

        x2 = x[:, :, :, 240:]     #  torch.Size([4, 3, 100, 240]
        x2 = self.backbone(x2)
        output2 = self.head2(x2)
        return output1, output2

def load_pretrain(model):
    model_dict = model.state_dict()
    dict_file = './result/pth/ShareNet-infov60-out0-4-20220607Bep0.pth'
    pretrained_dict = torch.load(dict_file)
    # print(pretrained_dict.keys())
    for k, v in pretrained_dict.items():
        if not(k in model_dict and v.shape == model_dict[k].shape):
            print("Attention! defined model can't matches perfectly with pretrained params")
            break
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

"""-------for debug-------"""
# from resnet_dataset import img_dataset_out2, img_dataset_out2_v2
# from torch.utils.data import DataLoader
# from torchvision import transforms
# if __name__ == '__main__':
#     # net = models.resnet18(pretrained=True)  #input_channel, output_nums
#     net = ShareResNet_Out2(3, 7)
#     # print(net)
#     root = "/disk2/dianzheng/lane_loc/lane_data/"
#     train_data = img_dataset_out2_v2(txt=root + 'txt_cnn/temp.txt', transform=transforms.ToTensor())
#     train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
#     for i, (images, labels1, labels2, img_file) in enumerate(train_loader):
#         print("images shape", images.shape)
#         print('labels1', labels1)
#         print('img file', img_file[0])
#         outputs1, outputs2 = net(images)
#         print("output shape", outputs1.shape)
#         exit()


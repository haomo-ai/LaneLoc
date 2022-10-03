#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, csv
import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model.share_resnet import *
from eval_util.eval_utils import *
from eval_util.eval_process import *

def parameters_parser():
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('-device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('-pth_name', default='./result/pth/ShareNet-infov60-out0-4-20220607Bep0.pth')
    parser.add_argument('-output_size', default=7)
    parser.add_argument('-img_dir', default='dataset/example/')
    parser.add_argument('-p_thread',default=[0.5, 0.7, 0.8])  # soft, medium, strict
    args = parser.parse_args()
    return args


def main(args): # CNN eval scene
    net = create_cnn(args) 
    eval_multi_imgs(net, args.img_dir)
    torch.cuda.empty_cache()

def create_cnn(args):
    net =  ShareResNet_Out2(3, args.output_size)
    net.load_state_dict(torch.load(args.pth_name, map_location='cpu'))
    return net

def preprocess(img):
    img = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
    x = img[100:270, 0:480, 0:3]
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = transforms.ToTensor()(x)
    x = x.unsqueeze(0)
    return x


def eval_multi_imgs(net, scene_dir):
    net.eval()
    img_lists = os.listdir(scene_dir)
    for img_file in img_lists:
        if not ('png' in img_file or 'jpg' in img_file):
            continue
        print(img_file)
        img_path = scene_dir + img_file
        img = cv2.imread(img_path)
        img = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = x[150:250, 0:480, 0:3]

        x = torch.FloatTensor(x)
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        with torch.no_grad():
            out1, out2 = net(x)
        save_one_img(out1, out2, img, scene_dir, img_file)

def save_one_img(out1, out2, img, scene_dir, img_file):
    left_str, right_str = str(out1[0].numpy()), str(out2[0].numpy())
    left_head, right_head = F.softmax(out1, dim=1), F.softmax(out2, dim=1)
    left_head, right_head = left_head[0], right_head[0]
    title = '          0      1       2       3       4  '
    left_str =  'Left : %.4f %.4f %.4f %.4f %.4f'%(left_head[0], left_head[1], left_head[2], left_head[3], left_head[4])
    right_str = 'Right: %.4f %.4f %.4f %.4f %.4f'%(right_head[0], right_head[1], right_head[2], right_head[3], right_head[4])
    img = cv2.putText(img, title, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    img = cv2.putText(img, left_str, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    img = cv2.putText(img, right_str, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    target_dir = './result/wrong_imgs/' + scene_dir.split('/')[-2] + '/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    cv2.imwrite( target_dir + img_file, img)

if __name__ == "__main__":
    main(parameters_parser())
    
    
    

    

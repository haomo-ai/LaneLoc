#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, csv
import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model.tu_model import *
from eval_util.eval_utils import *
from eval_util.eval_process import *

def parameters_parser():
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('-device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('-pth_name', default='./result/tu-pth0827/ShareNet-inTuData-out0-4-20220827ep11.pth')
    parser.add_argument('-output_size', default=7)
    parser.add_argument('-img_dir', default='dataset/TuSimple/lane_data/test_data/')
    parser.add_argument('-p_thread',default=[0.5, 0.7, 0.8])  # soft, medium, strict
    args = parser.parse_args()
    return args


def main(args): # CNN eval scene
    net = create_cnn(args) 
    eval_multi_imgs(net, args.img_dir, preprocess)
    torch.cuda.empty_cache()

def create_cnn(args):
    net =  ShareResNet_Out2(3, args.output_size)
    net.load_state_dict(torch.load(args.pth_name, map_location='cpu'))
    return net

def eval_multi_imgs(net, img_dir, preprocess):
    net.eval()
    for img_sequence in os.listdir(img_dir):
        for img_stamp in os.listdir(img_dir + '/' + img_sequence):
            img_file = img_dir + img_sequence + '/' + img_stamp + '/1.png'
            if not ('png' in img_file or 'jpg' in img_file):
                print('error, the input data is not image file')
                continue
            img = cv2.imread(img_file)
            x = preprocess(img)
            with torch.no_grad():
                out1, out2 = net(x)
            save_one_img(out1, out2, img, img_dir, img_file)

def preprocess(img):
    img = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
    x = img[100:270, 0:480, 0:3]
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = transforms.ToTensor()(x)
    x = x.unsqueeze(0)
    return x
    
if __name__ == "__main__":
    main(parameters_parser())
    
    
    

    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model.resnet_dataset import img_dataset_out2
from model.resnet_model import ShareResNet18_Out2
from model.share_resnet import ShareResNet_Out2
# from infer_util.infer_utils import *
from infer_util.infer_process import *


def parameters_parser():
    print('load parameters')
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('-device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('-num_class', default=7)
    parser.add_argument('-batch_size', default=256)
    parser.add_argument('-pth_name', default='./result/pth/ShareNet-infov60l-out0-4-20220702ep3.pth') #.onnx
    # parser.add_argument('-pth_name', default='./result/pth/ShareNet-inCuData-out0-4-20220628ep10.pth')
    # parser.add_argument('-eval_txt', default=['lane_data/txt_cnn/arrow.txt', 'lane_data/txt_cnn/buffer.txt', 
    #                                 'lane_data/txt_cnn/fishbone.txt', 'lane_data/txt_cnn/jam.txt', 
    #                                 'lane_data/txt_cnn/night.txt'])
    # parser.add_argument('-eval_txt', default=['lane_data/QA_cnn/8wide_lane.txt'])
    parser.add_argument('-eval_txt_dir', default='lane_data/QA_cnn/')
    parser.add_argument('-infer_res_dir', default='lane_data/infer_result/')
    parser.add_argument('-p_thread',default=[0.5, 0.7, 0.8])  # soft, medium, strict
    args = parser.parse_args()
    return args

def main(args): # CNN
    net = create_cnn(args)
    eval_data = args.data
    eval_data = [args.eval_txt_dir + a for a in os.listdir(args.eval_txt_dir)]
    for eval_txt in eval_data:
        data_loader = create_cnn_data(eval_txt, args.batch_size)
        infer_res = inference(args, net, data_loader, eval_txt, inference_cnn)
        save_infer_txt(infer_res, args.infer_res_dir, eval_txt)
        torch.cuda.empty_cache()
        exit()

def create_cnn(args):
    print('load model dict file')
    net =  ShareResNet_Out2(3, args.num_class)
    net.load_state_dict(torch.load(args.pth_name, map_location='cpu'))
    return net

def create_cnn_data(txt_file, batch_size):
    print('load eval txt file', txt_file)
    data_set = img_dataset_out2(txt=txt_file, transform=transforms.ToTensor())
    print('评测数目：',  data_set.__len__() )
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, num_workers=4)
    data_num = data_set.__len__()
    return data_loader

def inference_cnn(net, x):
    with torch.no_grad():
        y_hat_l, y_hat_r = net(x)
    return y_hat_l, y_hat_r

def save_infer_txt(infer_res, infer_dir, res_txt):
    txt = open(infer_dir + res_txt, 'w')
    for item in infer_res:
        # label = ' '.join(str(a) for a in item) + '\n'
        txt.write(item[0] + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + '\n')
    print('save', infer_dir, res_txt)

if __name__ == "__main__":
    main2(parameters_parser())
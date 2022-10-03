import os
from data_util.txt_utils import *
from tqdm import tqdm
import argparse
import numpy as np
import random
import cv2

def parameters_parser():
    parser = argparse.ArgumentParser(description='依据文件夹中的img图片, 生成txt文本')
    
    parser.add_argument('-train_dir_cu', default='dataset/CULane/train_data/')
    parser.add_argument('-valid_dir_cu', default='dataset/CULane/valid_data/')
    parser.add_argument('-test_dir_cu', default='dataset/CULane/test_data/')
    parser.add_argument('-lane_data_dir', default='dataset/CULane/lane_data/')
    parser.add_argument('-split_data_dir', default='dataset/CULane/list/test_split/')
    
    parser.add_argument('-train_txt', default='dataset/txt_files/train_data_cu.txt')
    parser.add_argument('-valid_txt', default='dataset/txt_files/valid_data_cu.txt')
    parser.add_argument('-test_txt', default='dataset/txt_files/test_data_cu.txt')
    args = parser.parse_args()
    return args


def main(args):
    data_sources = ['dataset/CULane/list/train.txt',
                    'dataset/CULane/list/val.txt',
                    'dataset/CULane/list/test.txt',]
    for txt_file in data_sources:
        print(txt_file)
        fp = open(txt_file, 'r')
        result_list = []
        for line in fp:
            line = line.strip('\n')
            line = line.rstrip()
            _, frame, mp4, img_name = line.split('/')
            # print(frame, mp4, name)
            img_path = args.lane_data_dir + frame + '/' + mp4 + '/' + img_name
            json_path = img_path.replace('.jpg', '.json')
            if not os.path.exists(json_path):
                continue
            left_index, right_index, scene, version = get_lane_index(json_path)
            result_list.append([img_path, left_index, right_index, scene, version])
        if 'train' in txt_file:
            generate_txt_files4(args.train_txt, result_list)
        elif 'val' in txt_file:
            generate_txt_files4(args.valid_txt, result_list)
        elif 'test' in txt_file:
            generate_txt_files1(args.test_txt, result_list)
            

def main2(args): # 分场景评测生成txt文件,用于Q2采集的特殊场景数据
    txt_list = os.listdir(args.split_data_dir)
    for txt_file in txt_list:
        fp = open(args.split_data_dir + txt_file, 'r')
        cu_scene = txt_file.split('.')[0]
        result_list = []
        for line in fp:
            line = line.strip('\n')
            line = line.rstrip()
            frame, mp4, img_name = line.split('/')
            # print(frame, mp4, name)
            img_path = args.lane_data_dir + frame + '/' + mp4 + '/' + img_name
            json_path = img_path.replace('.jpg', '.json')
            if not os.path.exists(json_path):
                continue
            left_index, right_index, scene, version = get_lane_index(json_path)
            result_list.append([img_path, left_index, right_index, scene, version])
        test_txt = 'dataset/txt_files/' + cu_scene + '.txt'
        generate_txt_files1(test_txt, result_list)


if __name__ == '__main__':
    main2(parameters_parser())
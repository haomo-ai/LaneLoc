import os
from data_util.txt_utils import *
from tqdm import tqdm
import argparse
import numpy as np
import random
import cv2

def parameters_parser():
    parser = argparse.ArgumentParser(description='依据文件夹中的img图片, 生成txt文本')
    parser.add_argument('-train_dir_tu', default='dataset/TuSimple/lane_data/train_data/')
    parser.add_argument('-valid_dir_tu', default='dataset/TuSimple/lane_data/valid_data/')
    parser.add_argument('-test_dir_tu', default='dataset/TuSimple/lane_data/test_data/')
    parser.add_argument('-train_txt', default='dataset/txt_files/train_data_tu.txt')
    parser.add_argument('-valid_txt', default='dataset/txt_files/valid_data_tu.txt')
    parser.add_argument('-test_txt', default='dataset/txt_files/test_data_tu.txt')
    args = parser.parse_args()
    return args

def main(args):
    '''---------------train txt---------------------'''
    data_dirs = [args.train_dir_tu, args.valid_dir_tu, args.test_dir_tu]
    for datadir in data_dirs:
        result_list = []
        for card_id in os.listdir(datadir):
            if not card_id.startswith('0'):
                continue
            print(".....processing data......",datadir, card_id)
            card_list = prapare_card_tu(datadir, card_id)
            result_list = result_list + card_list
        # generate_txt_files4(args.train_txt, train_label_list)
        if 'train' in datadir:
            generate_txt_files4(args.train_txt, result_list)
        elif 'val' in datadir:
            generate_txt_files4(args.valid_txt, result_list)
        elif 'test' in datadir:
            generate_txt_files1(args.test_txt, result_list)

# 从一个dataset dir中生成dataset
def prapare_card_tu(img_dir,  card_id):
    clip_lists = os.listdir(img_dir + card_id)
    clip_lists.sort()
    clip_path = img_dir + '/' + card_id                    
    result_list =  []
    for i in tqdm(range(len(clip_lists))):
        json_file = clip_path + '/' + clip_lists[i] + '/1.json'
        if not os.path.exists(json_file):
            continue
        left_index, right_index, scene, version = get_lane_index(json_file)
        img_path = clip_path + '/' +clip_lists[i] + '/1.png'
        result_list.append([img_path, left_index, right_index, scene, version])
    return result_list

if __name__ == '__main__':
    main(parameters_parser())
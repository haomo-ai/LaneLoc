import os
from data_util.txt_utils import *
from tqdm import tqdm
import argparse
import numpy as np
import random
import cv2

def parameters_parser():
    parser = argparse.ArgumentParser(description='依据文件夹中的img图片, 生成txt文本')
    parser.add_argument('-train_dir', default=['/disk2/lane_data/Q1resized_dataset/train_data/', '/img'])
    parser.add_argument('-test_dir', default=['/disk2/lane_data/Q1resized_dataset/test_data/', '/img'])

    parser.add_argument('-train_dir_tu', default=['dataset/TuSimple/train_data60/', ''])
    parser.add_argument('-test_dir_tu', default=['dataset/TuSimple/test_data60/', ''])

    parser.add_argument('-scene_jam_dir', default='/disk2/lane_data/scene_jam/')
    parser.add_argument('-dirty_data_dir', default='/disk2/lane_data/dirty_data/')
    parser.add_argument('-train_txt', default='dataset/txt_files/train_data60.txt')
    parser.add_argument('-test_txt', default='dataset/txt_files/test_data60.txt')
    args = parser.parse_args()
    return args

def main(args):
    '''---------------train txt---------------------'''
    train_label_list = []
    train_dirs = [args.train_dir]
    for train_dir in train_dirs:
        for card_id in os.listdir(train_dir[0]):
            if not (card_id.startswith('6') or card_id.startswith('0') or card_id.startswith('driver')):
                continue
            print(".....processing train card......", card_id)
            train_list = prapare_card(train_dir, args.dirty_data_dir, card_id)
            train_label_list = train_label_list + train_list
    generate_txt_files1(args.train_txt, train_label_list)

    '''---------------test txt--------------------'''
    test_label_list = []
    test_dirs = [args.test_dir]
    for test_dir in test_dirs:
        for card_id in os.listdir(test_dir[0]):
            if not (card_id.startswith('6') or card_id.startswith('0') or card_id.startswith('driver')):
                continue
            print(".....processing test card......", card_id)
            test_list = prapare_card(test_dir, args.dirty_data_dir, card_id)
            test_label_list = test_label_list + test_list
    generate_txt_files1(args.test_txt, test_label_list)

def main2(args): # 分场景评测生成txt文件,用于Q2采集的特殊场景数据
    root = '/data/lane_data/QAdata/'
    # root = '/disk2/lane_data/Open-Dataset/CULane-resized2/test_split/'
    test_label_list = []
    for scene in os.listdir(root):
        test_label_list = []
        test_dir = root + scene + '/'
        for card_id in os.listdir(test_dir):
            print(".....processing test card......", card_id)
            test_list = prapare_card([test_dir, '/img60'], args.dirty_data_dir, card_id)
            test_label_list = test_label_list + test_list
        test_txt = 'lane_data/QA_cnn/' + scene + '.txt'
    # test_txt = 'lane_data/QA_cnn/QAdata_eval.txt'
        generate_txt_files1(test_txt, test_label_list)

from data_util.city_list import *
# def parameters_parser():
#     parser = argparse.ArgumentParser(description='依据文件夹中的img图片, 生成test和traintxt文本')
#     parser.add_argument('-scenes_dir', default=['/disk2/lane_data/Q1cards/', '/img'])
#     parser.add_argument('-scenes_dir2', default=['/data/lane_data/Q2cards/', '/img60'])
#     parser.add_argument('-scenes_dir3', default=['/data/lane_data/special_scene_data/', '/img60'])
#     parser.add_argument('-scenes_dir4', default=['/disk2/lane_data/Q2NOH/', '/img60'])
#     parser.add_argument('-dirty_data_dir', default='/disk2/lane_data/dirty_data/')
#     parser.add_argument('-train_txt', default='dataset/txt_files/train_data60.txt')
#     parser.add_argument('-test_txt', default='dataset/txt_files/test_data60.txt')
#     args = parser.parse_args()
#     return args

def main3(args): # 直接从原始card dir中生成train_data和test_data
    test_label_list, train_label_list = [], []
    scenes_dirs = [args.scenes_dir, args.scenes_dir2, args.scenes_dir3, args.scenes_dir4]
    for scene_dir in scenes_dirs:
        print('==========', scene_dir)
        sceneList = os.listdir(scene_dir[0])
        if 'Q1cards' in scene_dir[0] or 'Q2cards' in scene_dir[0]:
                sceneList = ['']
        for scene in sceneList:
            cards_dir = scene_dir[0] + scene
            print(cards_dir)
            for i, card_id in enumerate(os.listdir(cards_dir)):
                if not card_id.startswith('6'):
                    continue
                if card_id in baoding_cards() or card_id in beijing_cards():
                    if random.random() > 0.5:
                        print('skip Q1 route card', card_id); continue
                print(".....processing card......",cards_dir, scene, card_id, i)
                img_dir = [cards_dir, scene_dir[1]]
                res_list = prapare_card(img_dir, args.dirty_data_dir, card_id)
                test_num = int(len(res_list) * 0.2)
                train_list, test_list = res_list[test_num:], res_list[:test_num]
                train_label_list = train_label_list + train_list
                test_label_list = test_label_list + test_list
    generate_txt_files4(args.train_txt, train_label_list)
    generate_txt_files4(args.test_txt, test_label_list)

if __name__ == '__main__':
    main(parameters_parser())
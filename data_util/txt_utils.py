#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from data_util.txt_utils import *
from tqdm import tqdm
import json

# 从一个dataset dir中生成 test 和 train dataset
def prapare_card(img_dir, dirty_data_dir, card_id):
    result_list =  []
    imgs_dir = img_dir[0] + card_id + img_dir[1]
    if not os.path.exists(imgs_dir):
        print(imgs_dir, 'not exist')
        return result_list
    img_name_list = os.listdir(imgs_dir)
    img_name_list.sort()
    err_list, static_list = get_err_list(dirty_data_dir, card_id), get_static_list(dirty_data_dir, card_id)
    dirty_list = err_list + static_list
    print("in this card, all imgs list:", len(img_name_list), "err list:", len(err_list), 'static list:', len(static_list))
    json_path = img_dir[0] + '/' + card_id + '/json/'                     # 标注json文件目录
    for img_file in tqdm(img_name_list):
        if img_file.split('.')[0] in dirty_list:
            continue
        json_file = os.path.join(json_path, img_file.replace("png", "json").replace("jpg", "json"))
        if not os.path.exists(json_file):
            print('json not found', json_file)
            continue
        left_index, right_index, scene, version = get_lane_index(json_file)
        img_file = os.path.join(imgs_dir, img_file)
        result_list.append([img_file, left_index, right_index, scene, version])
    return result_list

def get_err_list(dirty_data_dir, card):
    result = []
    paths = [dirty_data_dir + card + '/err_label_list.txt', 
            dirty_data_dir + card + '/err_list.txt',
            dirty_data_dir + card + '/static_wrong.txt']  # static_right.txt static_list.txt
    for path in paths:
        if not os.path.exists(path):
            continue
        err_list_file = open(path, 'r')
        for line in err_list_file:
            line = line.strip('\n')
            line = line.rstrip()
            line = line.split('.')[0]
            result.append(line)
    return result

def get_static_list(dirty_data_dir, card):
    result = []
    path = dirty_data_dir + card + '/static_right.txt'  # static_right.txt static_list.txt
    if not os.path.exists(path):
        return result
    err_list_file = open(path, 'r')
    for line in err_list_file:
        line = line.strip('\n')
        line = line.rstrip()
        line = line.split('.')[0]
        result.append(line)
    return result

#label_list = [jpg_file, left_index, right_index, scene]
# txt file : "jpg_file, left_index, total_lane" 生成用于评测的txt，生成一个txt
def generate_txt_files1(txt_path, label_list):
    threshold = 7 
    txt_file = open(txt_path, 'w')
    pass_num = 0
    print('用于生成评测集的数据', len(label_list))
    for label_item in label_list:
        jpg_file_path, left_index, right_index, scene, version = label_item
        img_id = jpg_file_path.split('/')[-1]
        left_index = 0 if left_index > 90 else left_index
        right_index = 0 if right_index > 90 else right_index
        total_lane = left_index + right_index -1
        if scene == 0 and total_lane > threshold:
            pass_num = pass_num + 1
            continue
        if scene == 92:   #车道总数变化
            left_index, right_index = 0, 0
        if scene in [90, 92, 94, 96, 97, 98, 99]: 
            pass_num = pass_num + 1
            continue
        if scene == 95: 
            # if left_index > 1000:
            #     left_index = left_index -1000
            # if right_index > 1000:
            #     right_index = right_index -1000
            if left_index > 90:
                left_index = 0
            if right_index > 90:
                right_index = 0
        if scene == 91:
            left_index, right_index = 0, 0
        txt_file.write('{} {} {} {}\n'.format(jpg_file_path, left_index, right_index, scene))   
    txt_file.close()
    print("save txt file", txt_path, "pass num", pass_num)

#label_list = [jpg_file, left_index, right_index, scene, version]
# txt file : "jpg_file, left_index, total_lane"
def generate_txt_files4(txt_path, label_list):
    print('用于生成训练集的数据0-4', len(label_list))
    threshold = 4
    txt_file = open(txt_path, 'w')
    for label_item in label_list:
        jpg_file_path, left_index, right_index, scene, version = label_item
        total_lane = left_index + right_index -1
        item_nums = 1
        if scene == 0:
            if(left_index > threshold):
                left_index = 0
                item_nums = 3
            if(right_index > threshold):
                right_index = 0
                item_nums = 3
        elif scene == 90: 
            if left_index > 80:
                left_index = 0
            if right_index > 80:
                right_index = 0
        elif scene == 91:
            left_index, right_index = 0, 0
        elif scene == 95:             # 跟上一帧相同
            if version == 15:
                continue                  
            if(left_index > 100):          
                left_index = 0
                item_nums = 3
            if(right_index > 100):
                right_index = 0
                item_nums = 3
            if(left_index > threshold):
                left_index = 0
            if(right_index > threshold):
                right_index = 0
        elif(scene == 93):                   # 严重堵车
            left_index, right_index = 0, 0
        else:
            continue
        for j in range(item_nums):
            txt_file.write('{} {} {}\n'.format(jpg_file_path, left_index, right_index))
    txt_file.close()
    print("save txt files", txt_path)


def get_lane_index(json_file_path):
    fp = open(json_file_path, 'r', encoding='utf8')
    json_data = json.load(fp)
    left_index = json_data['lane_localization']['lane_index_left']
    right_index = json_data['lane_localization']['lane_index_right']
    scene = json_data['lane_localization']['scene']
    if 'loc_version' in json_data.keys():
        version = int(json_data['loc_version'])
    else:
        version = 15
    return left_index, right_index, scene, version
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2, json
from tqdm import tqdm


def main():
    filename = '/disk2/dianzheng/LaneLocNet/tools/tusimple.mp4'
    path = '/disk2/dianzheng/LaneLocNet/result/tu_result/'
    img_dim = [(320, 180), (5, 15), (5, 30), (290, 10), (300, 15)]
    fps = 5
    videoWrite = cv2.VideoWriter(filename, # 参数：1 file name 2 编码器
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                fps, img_dim[0]) # 帧率, size
    for card in os.listdir(path):
        img_lists = os.listdir(path + card)
        print('processing....', card)
        img_lists.sort()
        for img_file in tqdm(img_lists):
            dirr = path + card + '/' + img_file 
            img = cv2.imread(dirr + '/1.jpg')
            img = cv2.resize(img, img_dim[0], interpolation=cv2.INTER_AREA)
            labels = get_lane_index(dirr + '/1.json')
            label_txt = ' '.join([str(t) for t in labels])
            img = cv2.putText(img, label_txt, img_dim[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            img = cv2.putText(img, 'test/' + card + '/' + img_file + '/1.jpg', img_dim[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            img = cv2.drawMarker(img, position=img_dim[3], color=(0,0,255), markerSize = 10, markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
            img = cv2.putText(img, str(fps), img_dim[4], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            videoWrite.write(img)
    videoWrite.release()
    print('end!')

def get_lane_index(json_file_path):
    fp = open(json_file_path, 'r', encoding='utf8')
    json_data = json.load(fp)
    left_index = json_data['lane_localization']['lane_index_left']
    right_index = json_data['lane_localization']['lane_index_right']
    scene = json_data['lane_localization']['scene']
    emergency_scene = json_data['lane_localization']['emergency_scene']
    stop_area = json_data['lane_localization']['stop_area']
    return [left_index, right_index, scene, 9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        -2000, stop_area]

if __name__ == '__main__':
    main()
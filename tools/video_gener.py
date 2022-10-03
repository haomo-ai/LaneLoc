#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2, json
from tqdm import tqdm


def main():
    filename = '/disk2/dianzheng/LaneLocNet/tools/cu_res.mp4'
    path = '/disk2/dianzheng/LaneLocNet/result/cu_result-0708/'
    img_dir = '/' 
    img_dim = [(480, 270), (10, 30), (10, 60), (400, 20), (410, 25)] #tu
    img_dim = [(960, 345), (10, 30), (10, 60), (900, 20), (910, 25)] #tu
    fps = 5
    videoWrite = cv2.VideoWriter(filename, # 参数：1 file name 2 编码器 3 帧率 4 size
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                fps, img_dim[0])
    for card in os.listdir(path):
        img_lists = os.listdir(path + card + img_dir)
        print('processing....', card)
        img_lists.sort()
        for img_file in tqdm(img_lists):
            img = cv2.imread(path + card + img_dir + img_file)
            # img = cv2.resize(img, img_dim[0], interpolation=cv2.INTER_AREA)
            # img = cv2.putText(img, card, img_dim[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            img = cv2.drawMarker(img, position=img_dim[3], color=(0,0,255), markerSize = 10, markerType=cv2.MARKER_TILTED_CROSS, thickness=1)
            img = cv2.putText(img, str(fps), img_dim[4], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            videoWrite.write(img)
    videoWrite.release()
    print('end!')

if __name__ == '__main__':
    main()
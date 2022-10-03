#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2, json
from tqdm import tqdm


def main():
    filename = 'cu_23.mp4'
    path = './train_data/driver_23_30frame/'
    lists = os.listdir(path)
    lists.sort()
    img_dim = [(410, 148), (5, 15), (5, 30), (380, 10), (390, 15)]
    fps = 5
    videoWrite = cv2.VideoWriter(filename, # 参数：1 file name 2 编码器
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                                fps, img_dim[0]) # 帧率, size
    for frame in tqdm(lists):
        img_lists = os.listdir(path + frame)
        img_lists.sort()
        for img_file in img_lists:
            if not 'jpg' in img_file:
                continue
            img = cv2.imread(path + frame + '/' + img_file)
            img = cv2.resize(img, img_dim[0], interpolation=cv2.INTER_AREA)
            labels = get_lane_index(path + frame + '/' + img_file.replace('jpg', 'json'))
            if labels is None:
                label_txt = 'Not marked'
            else:
                label_txt = ' '.join([str(t) for t in labels])
            img = cv2.putText(img, label_txt, img_dim[1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            file_name = path[13:] + frame + '/' + img_file
            img = cv2.putText(img, file_name, img_dim[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            img = cv2.drawMarker(img, position=img_dim[3], 
                                    color=(0,0,255), 
                                    markerSize = 10, 
                                    markerType=cv2.MARKER_TILTED_CROSS, 
                                    thickness=1)
            img = cv2.putText(img, str(fps), img_dim[4], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            videoWrite.write(img)
    videoWrite.release()
    print('end!')

def get_lane_index(json_file_path):
    if not os.path.exists(json_file_path):
        return None
    fp = open(json_file_path, 'r', encoding='utf8')
    json_data = json.load(fp)
    left_index = json_data['lane_localization']['lane_index_left']
    right_index = json_data['lane_localization']['lane_index_right']
    scene = json_data['lane_localization']['scene']
    emergency_scene = json_data['lane_localization']['emergency_scene']
    stop_area = json_data['lane_localization']['stop_area']
    return [left_index, right_index, scene, emergency_scene-2000, stop_area]

if __name__ == '__main__':
    main()
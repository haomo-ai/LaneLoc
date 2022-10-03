import os
import cv2
import shutil
import time
import csv


class Timer:
    def __init__(self):
        self.times = []
    def start(self):
        self.tik = time.time()
    def now_time(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class ShowProcess:
    def __init__(self, content_size=100, char_str='>'):
        self.content_size = content_size
        self.char_str = char_str
        self.char_long = 50
    def print_process(self, index, start_time):
        percentage = index / self.content_size
        print('\r',f'进度:[%-{self.char_long}s%.2f%%]耗时：%.1fs' % (
                  self.char_str * int(self.char_long * percentage), percentage * 100, time.time() - start_time), end='')

def clear_wrong_imgs():
    if os.path.exists('./result/wrong_imgs/'):
        shutil.rmtree('./result/wrong_imgs/')
    os.makedirs('./result/wrong_imgs/')

def print_overall_pr(overall_accum):
    print(overall_accum[2], overall_accum[1], overall_accum[0])
    print(f"**********overall evaluation**********\n",
          f"precision/recall is: {overall_accum[0] / overall_accum[1] * 100:.2f}%/"
          f"{overall_accum[0] / overall_accum[2] * 100:.2f}%")
          
def print_precision_recall(accumulator, name_str, file_name, data_file):
    record_file = open('./result/eval_csv/' + file_name + '.csv', 'a')
    f_csv = csv.writer(record_file)
    nonzero_total_sum = 0
    nonzero_predict_sum = 0
    nonzero_right_sum = 0
    print(f"**********{name_str} evaluation*************")
    f_csv.writerow([f"**********{name_str} evaluation*************"])
    f_csv.writerow(['evaluate data :' + data_file])
    f_csv.writerow(['label', '样本数量','预测数量','正确数量','未召回数量', 'precision', 'recall', 'inner_precision']) 
    for label, four_res in enumerate(accumulator):
        eval_item = []
        # print(four_res[0], four_res[1], four_res[2], four_res[3])
        print(f"label: {label}, {int(four_res[0])},{int(four_res[1])},{int(four_res[2])},{int(four_res[3])},", end="")
        eval_item += [str(label), str(int(four_res[0])), str(int(four_res[1])), str(int(four_res[2])), str(int(four_res[3]))]
        if label == 0:
            eval_item.append('None')
            if four_res[0] != 0:
                print(f' recall : {four_res[3] / four_res[0] * 100:.2f}%')
                eval_item.append(f'{four_res[3] / four_res[0] * 100:.2f}%')
            else:
                print(f" missing recall(data does not have truth label 0)")
                eval_item.append('None')
        else:
            if four_res[1] == 0:
                print(f' missing precision(data does not have predict label {label})', end=",")
                eval_item.append('None')
            else:
                print(f' precision: {four_res[2] / four_res[1] * 100:.2f}%', end=",")
                eval_item.append(f'{four_res[2] / four_res[1] * 100:.2f}%')
            if four_res[0] == 0:
                print(f' missing recall and precision_inner(data does not have truth label {label})')
                eval_item.append('None')
            else:
                print(f' recall : {four_res[2] / four_res[0] * 100:.2f}%', end=",")
                eval_item.append(f'{four_res[2] / four_res[0] * 100:.2f}%')
                if (four_res[0] - four_res[3]) == 0:
                    print(f'missing precision_inner(predict is all zero)')
                    eval_item.append('None')
                else:
                    print(f' precision_inner: {four_res[2] / (four_res[0] - four_res[3]) * 100:.2f}%')
                    eval_item.append(f'{four_res[2] / (four_res[0] - four_res[3]) * 100:.2f}%')
            nonzero_total_sum += four_res[0]
            nonzero_predict_sum += four_res[1]
            nonzero_right_sum += four_res[2]
        f_csv.writerow(eval_item)
    print('total', nonzero_total_sum, nonzero_predict_sum, nonzero_right_sum)
    f_csv.writerow(['total', nonzero_total_sum, nonzero_predict_sum, nonzero_right_sum])
    if nonzero_predict_sum != 0:
        total_precision = nonzero_right_sum / nonzero_predict_sum
        print(f'total_precision: {total_precision * 100:.2f}%')
        
    else:
        print('Attention: net predict results is all zero!!!!!!')
    if nonzero_total_sum != 0:
        total_recall = nonzero_right_sum / nonzero_total_sum
        print(f'total_recall: {total_recall * 100:.2f}%')
    else:
        print(f'Attention: truth label is all zero!!!!!!!')
    f_csv.writerow(['total precision: ','total recall: ' ])
    f_csv.writerow([f'{total_precision * 100:.2f}%', f'{total_recall * 100:.2f}%'])
    if name_str == "left/right fused":
        f_csv.writerow(["==================================================="])
    record_file.close()

def creat_record_file(args):
    file_name = args.pth_name.split('/')[-1][:-4] + '-' + str(len(args.test_data))
    record_file = open('./result/eval_csv/' + file_name + '.csv', 'w')
    f_csv = csv.writer(record_file)
    f_csv.writerow([args.pth_name])
    f_csv.writerow([ 'soft:' + str(args.p_thread[0]), 'medium:' + str(args.p_thread[1]), 'strict:' + str(args.p_thread[2])])
    record_file.close()
    return file_name


def save_wrong_img(img_file, predict_left, predict_right, label_left, label_right, left_head, right_head, color):
    # img_file = img_file.replace('/CULane-resized/test_data/', '/CULane-Dataset/').replace('png', 'jpg')
    wrong_img = cv2.imread(img_file)
    res_img = cv2.resize(wrong_img, (480, 270), interpolation=cv2.INTER_AREA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not isinstance(predict_left, int):
        predict_left = predict_left.item()
    # put_text以左上角为原点, img->shape:[h,w,c]
    img_paths = img_file.split('/')
    # res_img = cv2.putText(res_img, img_paths[-3] + '/' + img_paths[-1], (10, 265), font, 0.5, color[0], 2)
    truth = 'Truth: left ' + str(label_left.item()) + ', right ' + str(label_right.item())
    res_img = cv2.putText(res_img, truth,(10, 30), font, 0.7, color[0], 2)
    prediction = 'Prediction: left ' + str(predict_left)
    res_img = cv2.putText(res_img, prediction,(10, 60), font, 0.7, color[0], 2)
    left_str =  'Left : %.4f %.4f %.4f %.4f %.4f'%(left_head[0], left_head[1], left_head[2], left_head[3], left_head[4])
    right_str = 'Right: %.4f %.4f %.4f %.4f %.4f'%(right_head[0], right_head[1], right_head[2], right_head[3], right_head[4])
    res_img = cv2.putText(res_img, left_str, (12, 85), font, 0.5, color[1], 2)
    res_img = cv2.putText(res_img, right_str, (10, 110), font, 0.5, color[2], 2)
    # targt_dir = os.path.join('./result/wrong_imgs/', 'truth_' + str(label_left.item()))
    # targt_dir = os.path.join('./result/wrong_imgs/', 'predict_' + str(predict_right))
    targt_dir = os.path.join('./result/wrong_imgs/', img_paths[-3])
    if not os.path.exists(targt_dir):
        os.mkdir(targt_dir)
    cv2.imwrite(os.path.join(targt_dir, img_paths[-1]), res_img)
    # cv2.imshow('result image', res_img)
    # cv2.waitKey(1000)

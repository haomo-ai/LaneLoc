import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from eval_util.is_accuracy import *
from eval_util.eval_utils import *
import torch


def evaluate(args, net, data_loader, eval_file, record_file_name, inference_cnn):
    print(args.pth_name, eval_file, args.p_thread)
    l_accum, r_accum, overall_accum, l_r_accum = eval_one_txt(
        net, data_loader, 7, args.p_thread, args.device, args.batch_size, inference_cnn)
    # print_overall_pr(overall_accum)
    print_precision_recall(l_accum, "left", record_file_name, eval_file)
    print_precision_recall(r_accum, "right",record_file_name, eval_file)
    print_precision_recall(l_r_accum, "left/right fused", record_file_name, eval_file)

def eval_one_txt(net, data_loader, output_classes, p_thread, device, batch_size, infence):
    net.eval()
    net.to(device)
    cpu = torch.device("cpu")
    # accumulator 每个元素中包含label: {truth_label_total_nums, predict_label_total_nums, correct_predict_nums, predict_output_zero}
    left_accum = [Accumulator(4) for _ in range(output_classes)]
    right_accum = [Accumulator(4) for _ in range(output_classes)]
    l_r_accum = [Accumulator(4) for _ in range(output_classes)]
    overall_accum = Accumulator(3)
    for images, labels1, labels2, img_files, scenes in tqdm(data_loader):
        imgs = images.to(device)
        predicts_left, predicts_right = infence(net, imgs) 
        labels_left = labels1.to(cpu)
        labels_right = labels2.to(cpu)
        predicts_left = predicts_left.to(cpu) 
        predicts_right = predicts_right.to(cpu)
        confids_left = F.softmax(predicts_left, dim=1)
        confids_right = F.softmax(predicts_right, dim=1)
        cmp_left, cmp_right, indexs_left, indexs_right = is_accuracy_single(
                            confids_left, confids_right, labels_left, labels_right, p_thread, device)
        update_single_accum(left_accum, right_accum, cmp_left, cmp_right, indexs_left, 
                            indexs_right, labels_left, labels_right, img_files)
        cmp_multi, indexs_fusion, truths_nonzero = is_accuracy_multi_v2(
                            confids_left, confids_right, labels_left, labels_right, p_thread, device)
        update_fusion_accum(l_r_accum, overall_accum, cmp_multi, indexs_fusion, 
                            labels_left, truths_nonzero, img_files)
        save_eval_imgs(cmp_multi, cmp_left, cmp_right, confids_left, confids_right, 
                            labels_left, labels_right, truths_nonzero, indexs_fusion, indexs_right, img_files)
    return left_accum, right_accum, overall_accum, l_r_accum

def update_single_accum(l_accum, r_accum, cmp_left, cmp_right, indexs_left, indexs_right, labels_left, labels_right, img_file):
    # 评测单头的指标: 左
    # 评测数量, 预测数量, 正确数量, 预测为0数量}
    for j, (predict_label, truth_label, is_right) in enumerate(zip(indexs_left, labels_left, cmp_left)):
        try :
            l_accum[predict_label].add(0, 1, 0, 0)
            if not predict_label:
                l_accum[truth_label].add(0, 0, 0, 1)
            if is_right:
                l_accum[truth_label].add(1, 0, 1, 0)
            else:
                l_accum[truth_label].add(1, 0, 0, 0)
        except:
            print('left', img_file[j], truth_label)
    # 评测单头的指标: 右
    for j, (predict_label, truth_label, is_right) in enumerate(zip(indexs_right, labels_right, cmp_right)):
        try :
            r_accum[predict_label].add(0, 1, 0, 0)
            if not predict_label:
                r_accum[truth_label].add(0, 0, 0, 1)
            if is_right:
                r_accum[truth_label].add(1, 0, 1, 0)
            else:
                r_accum[truth_label].add(1, 0, 0, 0)
        except:
            print('right', img_file[j], truth_label)

def update_fusion_accum(l_r_accum, overall_accum, cmp_multi, indexs_fusion, labels_left, truths_nonzero, img_file):
    for j, (predict_label, truth_label, is_right) in enumerate(zip(indexs_fusion, labels_left, cmp_multi)):
        try :
            l_r_accum[predict_label].add(0, 1, 0, 0)
            if not predict_label:
                l_r_accum[truth_label].add(0, 0, 0, 1)
            if is_right:
                l_r_accum[truth_label].add(1, 0, 1, 0)
            else:
                l_r_accum[truth_label].add(1, 0, 0, 0)
        except:
            print('left_right', img_file[j], predict_label)
    predicts_nonzero = [1 for hat in indexs_fusion if hat > 0]
    overall_accum.add(sum(cmp_multi), sum(predicts_nonzero), sum(truths_nonzero))

def save_eval_imgs(cmp_multi, cmp_left, cmp_right, confids_left, confids_right, labels_left, labels_right, truths_nonzero, indexs_fusion, indexs_right, img_files):
    list_info = zip(cmp_multi, cmp_left, cmp_right, truths_nonzero, indexs_fusion, indexs_right)
    for j, (is_right, is_right_l, is_right_r, nz_y, y_hat, y_l) in enumerate(list_info):
        color_all, color_l, color_r = (255, 0, 0), (255, 0, 0), (255, 0, 0)
        # print(is_right, is_right_l, is_right_r, nz_y, y_hat)
        if not is_right and (y_hat or nz_y):
            color_all = (0, 0, 255)
        if not is_right_l and (y_hat):
            color_l = (0, 0, 255)
        if not is_right_r and (y_l):
            color_r = (0, 0, 255)
        color = [color_all, color_l, color_r]
        if  color_all == (0, 0, 255):#or color_r == (0, 0, 255):
            save_wrong_img(img_files[j], indexs_fusion[j], indexs_right[j], labels_left[j], labels_right[j], confids_left[j].tolist(), confids_right[j].tolist(), color)


def inference_cnn(net, x):
    with torch.no_grad():
        y_hat_l, y_hat_r = net(x)
    return y_hat_l, y_hat_r


def save_one_img(out1, out2, img, scene_dir, img_file):
    left_str, right_str = str(out1[0].numpy()), str(out2[0].numpy())
    left_head, right_head = F.softmax(out1, dim=1), F.softmax(out2, dim=1)
    left_head, right_head = left_head[0], right_head[0]
    title = '     0    1      2      3      4  '
    left_str =  'Left : %.4f %.4f %.4f %.4f %.4f'%(left_head[0], left_head[1], left_head[2], left_head[3], left_head[4])
    right_str = 'Right: %.4f %.4f %.4f %.4f %.4f'%(right_head[0], right_head[1], right_head[2], right_head[3], right_head[4])
    img = cv2.putText(img, title, (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    img = cv2.putText(img, left_str, (12, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    img = cv2.putText(img, right_str, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    target_dir = './result/wrong_imgs/' + scene_dir.split('/')[-2] + '/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    img_file = img_file.replace('/', '-')
    print(target_dir + img_file)
    cv2.imwrite(target_dir + img_file, img)
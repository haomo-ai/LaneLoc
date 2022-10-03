import torch


def is_accuracy_multi_v2(y_hat1, y_hat2, y1, y2, p_thresh, device):
    p_thresh1 = 0.95
    cmp = []
    y_hat_res, y_nonzero = [], []
    # y_hat1, y_hat2 = F.softmax(y_hat_l, dim=1)， F.softmax(y_hat_r, dim=1)
    for y_hat1_item, y_hat2_item, label_l, label_r in zip(y_hat1, y_hat2, y1, y2):
        idx_left = y_hat1_item.argmax(dim=0)
        idx_right = y_hat2_item.argmax(dim=0)
        p_left = y_hat1_item[idx_left]
        p_right = y_hat2_item[idx_right]
        if label_l == 0 or label_r == 0:
            total_lane = label_l + label_r
            if label_l == 0 and idx_left == 0:
                lane_idx = 0
            elif label_l == 0 and idx_left != 0:
                lane_idx = total_lane - idx_right if idx_right <= total_lane else 0
            elif label_r == 0 and idx_right == 0:
                lane_idx = idx_left if idx_left <= total_lane else 0           #!!!!!!!!!!????TODO
            elif label_r == 0 and idx_right != 0:
                lane_idx = idx_left if idx_left <= total_lane else 0
        else:
            total_lane = label_l + label_r - 1
            if idx_left == 0:
                if idx_right == 0 or idx_right == 6:
                    lane_idx = 0
                else:
                    lane_idx = total_lane + 1 - idx_right if idx_right <= total_lane else 0
            elif idx_left >0 and idx_left <= total_lane: # 左编号：1 —— 车道总数
                if idx_right == 0 or idx_right > total_lane:
                    lane_idx = idx_left
                else:                                    # 右编号也是1-5
                    lane_idx = idx_left if (p_left > p_right) else (total_lane + 1 - idx_right)
                    if idx_right > 3 and p_left > p_thresh1:
                        lane_idx = idx_left
                    if idx_left > 3 and p_right > p_thresh1:
                        lane_idx = total_lane + 1 - idx_right
            elif idx_left > total_lane: # 左编号：车道总数 —— 6
                if idx_right >0 and idx_right <= total_lane: # 左编号：1 —— 车道总数
                    lane_idx = total_lane + 1 - idx_right
                else:
                    lane_idx = 0   
        y_hat_res.append(lane_idx)
        if label_r or label_l:
            y_nonzero.append(1)
        else:
            y_nonzero.append(0)
        if lane_idx == 0 and label_l != 0:
            cmp.append(0)
        else:
            cmp.append(int(lane_idx == label_l))
    y_hat_res = torch.tensor(y_hat_res).to(device)
    return cmp, y_hat_res, y_nonzero

def is_accuracy_multi_unseen(y_hat1, y_hat2, y1, y2, p_thresh, device):
    p_thresh1 = 0.95
    cmp = []
    y_hat_res, y_nonzero = [], []
    # y_hat1, y_hat2 = F.softmax(y_hat_l, dim=1)， F.softmax(y_hat_r, dim=1)
    for y_hat1_item, y_hat2_item, label_l, label_r in zip(y_hat1, y_hat2, y1, y2):
        idx_left = y_hat1_item.argmax(dim=0)
        idx_right = y_hat2_item.argmax(dim=0)
        p_left = y_hat1_item[idx_left]
        p_right = y_hat2_item[idx_right]
        total_lane = label_l + label_r - 1
        if idx_left == 0:
            if idx_right == 0 or idx_right == 6:
                lane_idx = 0
            else:
                lane_idx = total_lane + 1 - idx_right if idx_right <= total_lane else 0
        elif idx_left == 6:  # unseen
            if idx_right == 0 or idx_right == 6:
                lane_idx = 0
            else:
                lane_idx = total_lane + 1 - idx_right if idx_right <= total_lane else 0 
        elif idx_left >0 and idx_left <= total_lane: # 左编号：1 —— 车道总数
            if idx_right == 0 or idx_right > total_lane:
                lane_idx = idx_left
            else:                                    # 右编号也是1-5
                lane_idx = idx_left if (p_left > p_right) else (total_lane + 1 - idx_right)
                if idx_right > 3 and p_left > p_thresh1:
                    lane_idx = idx_left
                if idx_left > 3 and p_right > p_thresh1:
                    lane_idx = total_lane + 1 - idx_right
        elif idx_left > total_lane and idx_left < 6: # 左编号：车道总数 —— 6
            if idx_right >0 and idx_right <= total_lane: # 左编号：1 —— 车道总数
                lane_idx = total_lane + 1 - idx_right
            else:
                lane_idx = 0   
        y_hat_res.append(lane_idx)
        if label_r or label_l:
            y_nonzero.append(1)
        else:
            y_nonzero.append(0)
        if lane_idx == 0:
            cmp.append(0)
        else:
            cmp.append(int(lane_idx == label_l))
    y_hat_res = torch.tensor(y_hat_res).to(device)
    return cmp, y_hat_res, y_nonzero

def is_accuracy_single(p_l_all, p_r_all, y_l_all, y_r_all, p_thread, device):
    # p_l_all， p_r_all = F.softmax(y_hat_l, dim=1)， F.softmax(y_hat_r, dim=1)
    strict_p = p_thread[2]
    # TODO:目前真值数据如果用95的时候需要修改data_load类
    cmp_l, cmp_r = [], []
    y_hat_l_res, y_hat_r_res = [], []
    for i, (idx_l, idx_r) in enumerate(zip(p_l_all.argmax(dim=1), p_r_all.argmax(dim=1))):
        p_l_max = p_l_all[i][idx_l] # 预测概率
        p_r_max = p_r_all[i][idx_r]
        y_r = y_r_all[i]            # 真值
        y_l = y_l_all[i]
        #  # 将y>5的部分即包含特殊场景变为0
        # if y_r not in physical_lane_idx:
        #     y_r = 0
        # if y_l not in physical_lane_idx:
        #     y_l = 0
        # single output evaluation:
        if p_l_max < strict_p:
            y_hat_l_res.append(0)
        else:
            y_hat_l_res.append(idx_l)
            
        if p_r_max < strict_p:
            y_hat_r_res.append(0)
        else:
            y_hat_r_res.append(idx_r)
        # # 融合的时候再变0
        # if idx_r not in physical_lane_idx:
        #     idx_r = 0
        # if idx_l not in physical_lane_idx:
        #     idx_l = 0
    # 单头输出结果分别与真值比较
    for predict_l, y_l in zip(y_hat_l_res, y_l_all):
        if predict_l == y_l and predict_l:
            cmp_l.append(1)
        else:
            cmp_l.append(0)
    for predict_r, y_r in zip(y_hat_r_res, y_r_all):
        if predict_r == y_r and predict_r:
            cmp_r.append(1)
        else:
            cmp_r.append(0)
    return cmp_l, cmp_r, y_hat_l_res, y_hat_r_res

# 根据无95计算准确率版本
# physical_lan_idx = [1,2,3,4,5]
def is_accuracy_multi(p_l_all, p_r_all, y_l_all, y_r_all, p_thread, device):
    soft_p, medium_p, strict_p = p_thread[0], p_thread[1], p_thread[2]
    # p_l_all， p_r_all = F.softmax(y_hat_l, dim=1)， F.softmax(y_hat_r, dim=1)
    # TODO:目前真值数据如果用95的时候需要修改data_load类
    y_hat_res_all, y_nonzero = [], []  # 记录神经网络预测结果
    cmp_multi = []
    for i, (idx_l, idx_r) in enumerate(zip(p_l_all.argmax(dim=1), p_r_all.argmax(dim=1))):
        p_l_max = p_l_all[i][idx_l] # 预测概率
        p_r_max = p_r_all[i][idx_r]
        y_r = y_r_all[i]            # 真值
        y_l = y_l_all[i]
        # post-output evaluation:
        total_lane_valid = True
        total_lane_idx = y_r + y_l - 1
        if y_r > 5 or y_l > 5:
            total_lane_valid = True
            total_lane_idx =-1   # 总车道数
        # 这里代码再让车道序号变0，防止影响单头计算
        if idx_r > total_lane_idx:
            idx_r = 0
        if idx_l > total_lane_idx:
            idx_l = 0
        y_hat_res = 0  # 存储神经网络融合预测结果
        # 标注总车道数有效时
        if total_lane_valid:
            if p_l_max > strict_p and p_r_max > strict_p and idx_l != 0 \
                    and idx_r != 0 and idx_l != (total_lane_idx + 1 - idx_r):
                y_hat_res = 0
            elif p_l_max > soft_p and p_r_max > soft_p and idx_l != 0 \
                    and idx_r != 0 and idx_l == (total_lane_idx + 1 - idx_r):
                y_hat_res = idx_l
            elif p_l_max > strict_p and (p_r_max < medium_p or idx_r == 0):
                y_hat_res = idx_l
            elif p_r_max > strict_p and (p_l_max < medium_p or idx_l == 0):
                y_hat_res = total_lane_idx + 1 - idx_r
            else:
                y_hat_res = 0
            if y_hat_res == 0:
                cmp_multi.append(0)
            else:
                cmp_multi.append(int(y_hat_res == y_l))
        # 总车道数无效，此时只能根据左右编号概率谁大信谁的原则来得到融合结果
        else:
            if p_l_max > strict_p and p_r_max > strict_p:
                if p_l_max > p_r_max:
                    y_hat_res = idx_l
                    if y_l and idx_l == y_l:
                        cmp_multi.append(1)
                    else:
                        cmp_multi.append(0)
                else:
                    y_hat_res = idx_r
                    if y_r and idx_r == y_r:
                        cmp_multi.append(1)
                    else:
                        cmp_multi.append(0)
            elif p_l_max > strict_p:
                y_hat_res = idx_l
                if y_l and idx_l == y_l:
                    cmp_multi.append(1)
                else:
                    cmp_multi.append(0)
            elif p_r_max > strict_p:
                y_hat_res = idx_r
                if y_r and idx_r == y_r:
                    cmp_multi.append(1)
                else:
                    cmp_multi.append(0)
        y_hat_res_all.append(y_hat_res)
        if y_r or y_l:
            y_nonzero.append(1)
        else:
            y_nonzero.append(0)
    return cmp_multi, y_hat_res_all, y_nonzero
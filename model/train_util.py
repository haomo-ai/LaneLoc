#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# source ~/.bashrc

import time
import argparse
import csv
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model.learn_rate import CosineAnnealingLRWarmup
import math

def train_and_save(args, model, train_loader, valid_loader):
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)
    # wampup_lr_scheduler = CosineAnnealingLRWarmup(optimizer,  T_max=10,
    #                                             eta_min=1.0e-6, last_epoch=-1,
    #                                             warmup_steps=5, warmup_start_lr=1.0e-5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_acc, best_r_acc = 0.0, 0.0
    best_val_loss = 0.0
    patience = 10
    for epoch in range(args.epoches):
        print('Epoch: {}, training'.format(epoch))
        train_loss = train_epoch(args, model, optimizer, criterion, train_loader)
        print('Epoch: {}. train_loss: {:.5f}.'.format(epoch, train_loss))
        exp_lr_scheduler.step()
        # wampup_lr_scheduler.step()
        print(optimizer.param_groups[0]['lr'])
        valid_loss, epoch_acc = eval_model(args, epoch, model, criterion, valid_loader)
        best_val_loss = valid_loss if epoch == 0 else best_val_loss
        save_record(args.pth_name.split('/')[-1][:-4], [epoch], [train_loss, valid_loss], epoch_acc)
        if (epoch_acc[0] > best_acc or epoch_acc[2] > best_r_acc or valid_loss < best_val_loss):
            best_acc = epoch_acc[0] if epoch_acc[0] > best_acc else best_acc
            best_r_acc = epoch_acc[2] if epoch_acc[2] > best_r_acc else best_r_acc
            pth_name = args.pth_name.replace('.pth',  'ep' + str(epoch) + '.pth')
            print(pth_name)
            torch.save(model.state_dict(), pth_name)
            save_record(args.pth_name.split('/')[-1][:-4], [epoch], [pth_name], [])
    print('Training complete')
    print('Best val Acc: {:4f}'.format(best_acc))

def train_epoch(args, model, optimizer, criterion, train_loader):
    model.train()
    epoch_losses = []
    for images, labels1, labels2, _, _ in tqdm(train_loader):
        images, labels1, labels2 = images.to(args.device), labels1.to(args.device), labels2.to(args.device)
        # images = images.requires_grad_()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs1, outputs2 = model(images)
            loss = args.loss_ratio[0] * criterion(outputs1, labels1) + args.loss_ratio[1] * criterion(outputs2, labels2)
            loss.backward()
            optimizer.step()
        epoch_losses.append(loss.item())
    epoch_loss = np.average(epoch_losses)
    return epoch_loss

def eval_model(args, epoch, model, criterion, valid_loader):
    model.eval()
    correct, correct_l, correct_r, total = 0, 0, 0, 0
    valid_losses = []
    print('Epoch: {}, eval'.format(epoch))
    for images, labels1, labels2, _, _ in tqdm(valid_loader):                 # Iterate through test dataset
        images, labels1, labels2 = images.to(args.device), labels1.to(args.device), labels2.to(args.device)
        outputs1, outputs2 = model(images) 
        _, predicted1 = torch.max(outputs1, 1)
        _, predicted2 = torch.max(outputs2, 1)
        total += labels1.size(0)                        # Total number of labels
        right_prediction = 0 
        for j in range(labels1.size(0)):
            if(predicted1[j].item() == labels1[j].item() or predicted2[j].item() == labels2[j].item()):  
                right_prediction += 1
        correct += right_prediction         # Total correct predictions
        correct_l += (predicted1 == labels1).sum()
        correct_r += (predicted2 == labels2).sum()  
        loss = args.loss_ratio[0] * criterion(outputs1, labels1)  + args.loss_ratio[1] * criterion(outputs2, labels2)
        valid_losses.append(loss.item())
    valid_loss = np.average(valid_losses)
    print('Epoch:{}, valid_loss:{:.5f}'.format(epoch, valid_loss))
    accuracy_l_r, accuracy_l, accuracy_r = correct / total, correct_l / total, correct_r / total
    accuracy = [accuracy_l_r, accuracy_l.item(), accuracy_r.item()]
    print('Accuracy total:{}, left:{}, right:{}'.format(accuracy_l_r, accuracy_l, accuracy_r))
    return valid_loss, accuracy

def creat_record_file(args, tain_num, valid_num):
    file_name, description = args.pth_name.split('/')[-1][:-4], args.train_descript
    train_record = open('./result/train_csv/' + file_name + '.csv', 'w')
    f_csv = csv.writer(train_record)
    f_csv.writerow([description])
    f_csv.writerow(['learning rate', args.lr])
    f_csv.writerow(['loss ratio', args.loss_ratio])
    f_csv.writerow(['output size', args.output_size])
    f_csv.writerow(['batch size', args.batch_size])
    f_csv.writerow(['train data number', tain_num])
    f_csv.writerow(['valid data number', valid_num])
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f_csv.writerow(['begin time', timestr])
    f_csv.writerow(['epoch', 'train_loss', 'valid_loss', 'accuracy_l/r', 'accuracy_l', 'accuracy_r'])
    train_record.close()

def save_record(file_name, epoch, loss, accuracy):
    train_record = open('./result/train_csv/' + file_name + '.csv', 'a')
    f_csv = csv.writer(train_record)
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f_csv.writerow(epoch + loss + accuracy + [timestr])
    train_record.close()


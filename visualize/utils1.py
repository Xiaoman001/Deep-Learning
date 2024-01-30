# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:00:17 2019
utils.py
@author: AORUS
"""
import numpy as np
import os
# import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import numpy as np
from PIL import Image
import pylab
from pylab import *

import torch


def visualization(test_id, gt, pt, H, im_file, frame, save_path):

    gt_t = torch.from_numpy(gt)
    gt_t = torch.cat((gt_t, torch.ones([gt_t.size(0), gt_t.size(1), 1])), 2).transpose(1, 2)
    pt_t = torch.from_numpy(pt)
    pt_t = torch.cat((pt_t, torch.ones([pt_t.size(0), pt_t.size(1), 1])), 2).transpose(1, 2)
    pt_t = pt_t[-13:]

    img = mpimg.imread(im_file)
    plt.imshow(img)
    gt_pixel = torch.matmul(H, gt_t)
    gt_pixel1 = gt_pixel.transpose(1, 2).numpy()
    pt_pixel = torch.matmul(H, pt_t)
    pt_pixel1 = pt_pixel.transpose(1, 2).numpy()
    #for i in range(gt_pixel1.shape[1]):
    for i in [2,3]:
        x1 = gt_pixel1[:, i, 0] / gt_pixel1[:, i, 2]
        y1 = gt_pixel1[:, i, 1] / gt_pixel1[:, i, 2]
        x2 = pt_pixel1[:, i, 0] / pt_pixel1[:, i, 2]
        y2 = pt_pixel1[:, i, 1] / pt_pixel1[:, i, 2]

        if test_id == 0 or test_id == 1:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # x1, x2 = x1 - 15, x2 - 15
        # y1, y2 = y1 - 20, y2 - 20
        x1, y1, x2, y2 = x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist()
        # observed trajectory
        plt.plot(x1[:8], y1[:8], "b-", lw=2.5)
        # ground truth
        plt.plot(x1[7:], y1[7:], "b--", lw=2.5)
        # prediction trajectory
        plt.plot(x2[-13:], y2[-13:], "r--", lw=2.5)

    plt.xticks([])
    plt.yticks([])
    # 添加标题，显示绘制的图像
    plt.title('Frame:' + str(frame))
    plt.savefig(save_path)
    plt.show()

    plt.close()


def visualize(test_id, frame, gt, pt):

    H_dirs = ['./datasets/homography/eth.txt', './datasets/homography/hotel.txt',
              './datasets/homography/zara.txt', './datasets/homography/zara.txt',
              './datasets/homography/univ.txt']

    image_path = './datasets/' + str(test_id) + '/images/'
    
    if test_id == 0:
        frame = frame + 48
    else:
        frame = frame + 80

    im_file = image_path + str(frame) + '.jpg'
    H = numpy.loadtxt(H_dirs[test_id]).astype(np.float32)
    H_t = torch.pinverse(torch.from_numpy(H))
    save_path = './datasets/' + str(test_id) + '/save_images/'
    visualization(gt, pt, H_t, im_file, frame, save_path)


def load2pixel(file_name, H, iftrans=False):
    # load函数从.npy文件中加载原始像素坐标
    gt = np.load(file_name)
    print("gt",gt)  #gt(20,9,2)  pt(19,9,2)  9*2矩阵，2表示x，y，9表示9个人
    print(gt.shape)
    # 检查维度
    if gt.ndim == 3:
        if iftrans:  # x,y通道互换
            temp = gt[:, :, 0].copy()
            gt[:, :, 0], gt[:, :, 1] = gt[:, :, 1], temp
        ones = np.ones((gt.shape[0], gt.shape[1], 3))
        # print("1", ones)  #(20,9,3)
        ones[:, :, :2] = gt   #把gt复制进去
        # print("2",ones)  #(20,9,3)
        gt_t = np.transpose(ones, (0, 2, 1))
        print("gt_t",gt_t)  #(20,3,9)，实质是每个矩阵的转置  (20,9,3)变(20,3,9)
        gt_pixel = np.matmul(H, gt_t)   # 矩阵乘法，通过齐次坐标变换矩阵H  (20,3,9)
        gt_pixel = gt_pixel.transpose(0, 2, 1)  # 转置，实质是每个矩阵的转置  (20,3,9)变 (20,9,3)
        # print("gt_pixel", gt_pixel)

        gx = gt_pixel[:, :, 0] / gt_pixel[:, :, 2]   # (20,9)
        gy = gt_pixel[:, :, 1] / gt_pixel[:, :, 2]   # (20,9)
        print("gx",gx)
        print("yuan",gt_pixel[:, :, 0])

    elif gt.ndim == 4:
        ones = np.ones((gt.shape[0], gt.shape[1], gt.shape[2], 3))
        ones[:, :, :, :2] = gt
        gt_t = np.transpose(ones, (0, 1, 3, 2))
        gt_pixel = np.matmul(H, gt_t)
        gt_pixel = gt_pixel.transpose(0, 1, 3, 2)
        gx = gt_pixel[:, :, :, 0] / gt_pixel[:, :, :, 2]
        gy = gt_pixel[:, :, :, 1] / gt_pixel[:, :, :, 2]

    return gx, gy


def load2meter(file_name):
    gt = np.load(file_name)
    gx, gy = gt[..., 0], gt[..., 1]
    return gx, gy



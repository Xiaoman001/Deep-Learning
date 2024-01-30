# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:00:17 2019
utils.py
@author: AORUS
"""

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import numpy as np
from PIL import Image
# import pylab
# from pylab import *

import torch


def get_frames(filename):

    file_path = os.path.join(filename, 'true_pos_.csv')
    data = np.genfromtxt(file_path, delimiter=',')
    # frameslist = np.unique(data[0, :]).tolist()
    frames = data[0, :]
    frames = np.unique(frames)
    return frames


def get_images(video_path, image_path, frames):
    vc = cv2.VideoCapture(video_path)
    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        if c in frames:
            cv2.imencode('.jpg', frame)[1].tofile(image_path + str(c) + '.jpg')
        c = c + 1
    vc.release()


def prepare(filename, video_path, image_path):
    frames = get_frames(filename)
    get_images(video_path, image_path, frames)


def visualization(test_id, gt, pt, H, im_file, frame, save_path):

    gt_t = torch.from_numpy(gt)
    gt_t = torch.cat((gt_t, torch.ones([gt_t.size(0), gt_t.size(1), 1])), 2).transpose(1, 2)
    pt_t = torch.from_numpy(pt)
    pt_t = torch.cat((pt_t, torch.ones([pt_t.size(0), pt_t.size(1), 1])), 2).transpose(1, 2)
    pt_t = pt_t[-12:]

    img = mpimg.imread(im_file)
    plt.imshow(img)
    gt_pixel = torch.matmul(H, gt_t)
    gt_pixel1 = gt_pixel.transpose(1, 2).numpy()
    pt_pixel = torch.matmul(H, pt_t)
    pt_pixel1 = pt_pixel.transpose(1, 2).numpy()

    for i in range(gt_t.size(2)):
        x1 = gt_pixel1[:, i, 0] / gt_pixel1[:, i, 2]
        y1 = gt_pixel1[:, i, 1] / gt_pixel1[:, i, 2]
        x2 = pt_pixel1[:, i, 0] / pt_pixel1[:, i, 2]
        y2 = pt_pixel1[:, i, 1] / pt_pixel1[:, i, 2]

        if test_id == 0 or test_id == 1:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        x1, y1, x2, y2 = x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist()
        # observed trajectory
        plt.plot(x1[:8], y1[:8], "b-", lw=2.5)
        # ground truth
        plt.plot(x1[8:], y1[8:], "y--", lw=2.5)
        # prediction trajectory
        plt.plot(x2[-12:], y2[-12:], "r--", lw=2.5)

    plt.xticks([])
    plt.yticks([])
    # 添加标题，显示绘制的图像
    plt.title('Frame:' + str(frame))
    plt.savefig(save_path)
    plt.draw()
    plt.pause(1)
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
    save_path = './datasets/' + str(test_id) + '/save_images/'
    save_im = save_path + str(frame) + '.png'
    H = numpy.loadtxt(H_dirs[test_id]).astype(np.float32)
    H_t = torch.pinverse(torch.from_numpy(H))

    visualization(test_id, gt, pt, H_t, im_file, frame, save_im)


def csv2txt(path1, path2):
    data = np.genfromtxt(path1, delimiter=',')
    data = data.transpose(1,0)
    print(data.shape)
    np.savetxt(path2, data, fmt='%0.2f')



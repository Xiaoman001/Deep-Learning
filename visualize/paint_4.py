# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:00:17 2019
data_preprocess.py
@author: AORUS
"""
from utils1 import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

plt.ion()

H_dirs = ['./homography/eth.txt', './homography/hotel.txt',
          './homography/zara.txt', './homography/zara.txt',
          './homography/univ.txt']

# SRAI-LSTM-UNIV; SR-LSTM-UNIV; S-GAN-UNIV
#  eth 4793; hotel 10570; zara 1680; univ 4360
pixel = [4763, 10550, 1680, 1680, 4360]

test_id = 4

frame_ = pixel[test_id]

original_frame_ = frame_

H_ = torch.from_numpy(numpy.loadtxt(H_dirs[test_id]).astype(np.float32))
H = np.linalg.pinv(H_)


if test_id == 0:
    frame = frame_ - 42
else:
    frame = frame_ - 70

# 读取像素
gt_text1 = './batchs/' + str(test_id) + '/batch_sralstm/0_' + str(frame) + 'gt.npy'
pt_text1 = './batchs/' + str(test_id) + '/batch_sralstm/0_' + str(frame) + 'pt.npy'
gt_text2 = './batchs/' + str(test_id) + '/batch_srlstm/1_' + str(frame) + 'gt.npy'
pt_text2 = './batchs/' + str(test_id) + '/batch_srlstm/1_' + str(frame) + 'pt.npy'
gt_text4 = './batchs/' + str(test_id) + '/batch_sgan/sgan_' + str(frame) + '_gt.npy'
pt_text4 = './batchs/' + str(test_id) + '/batch_sgan/sgan_' + str(frame) + '_pt.npy'

gx1, gy1 = load2pixel(gt_text1, H)
px1, py1 = load2pixel(pt_text1, H)

gx2, gy2 = load2pixel(gt_text2, H, iftrans=True)
px2, py2 = load2pixel(pt_text2, H, iftrans=True)


gx4, gy4 = load2pixel(gt_text4, H)
px4, py4 = load2pixel(pt_text4, H)

# 多维矩阵，第一个数字表示批次
print(gx1.shape, gx2.shape, gx4.shape)
print(gx1.shape[-1])
Observe = 8
Predict = 12

for j in range (Observe):
    # 消除定义背景白色
    plt.figure(figsize=(7.2, 5.76))

    im_file = './imgs/' + str(test_id) + '/' + str(frame) + '.jpg'
    img = mpimg.imread(im_file)
    plt.imshow(img)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    plt.plot([], [], "r-", lw=3.0, label='Observed')

    for i in range(gx1.shape[-1]):
        if i in [4, 5]:
            # 第8到最后一行的数据
            gx_o = gx1[0:j+1, i]
            gy_o = gy1[0:j+1, i]
            plt.plot(gx_o, gy_o, "r-", lw=1.5)


    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right')

    # 每0.4s暂停一下
    plt.pause(0.4)

    if test_id == 0:
        frame += 6
    else:
        frame += 10

    plt.show()


for k in range (Predict):

    # 消除定义背景白色
    plt.figure(figsize=(7.2, 5.76))

    im_file = './imgs/' + str(test_id) + '/' + str(frame_) + '.jpg'
    img = mpimg.imread(im_file)
    plt.imshow(img)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    plt.plot([], [], "r-", lw=3.0, label='Observed')
    plt.plot([], [], "r--", lw=3.0, label='Ground truth')
    plt.plot([], [], "g--", lw=3.0, label='SRAI-LSTM')
    plt.plot([], [], "y--", lw=3.0, label='SR-LSTM')
    # plt.plot([], [], "b--", lw=3.0, label='S-GAN')

    for i in range(gx1.shape[-1]):
        if i in [2, 3]:
            plt.plot(px4[7:(8+k+1), i], py4[7:(8+k+1), i], "b--", lw=1.5)

        if i in [4, 5]:
            # 第8到最后一行的数据
            gx_o = gx1[:8, i]
            gy_o = gy1[:8, i]
            gx_p = gx1[7:(8+k+1), i]
            gy_p = gy1[7:(8+k+1), i]
            plt.plot(gx_o, gy_o, "r-", lw=1.5)
            plt.plot(gx_p, gy_p, "r--", lw=1.5)
            plt.plot(px1[7:(8+k+1), i], py1[7:(8+k+1), i], "g--", lw=1.5)
            plt.plot(px2[7:(8+k+1), i], py2[7:(8+k+1), i], "y--", lw=1.5)


    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right')

    # 每0.4s暂停一下
    plt.pause(0.4)

    if test_id == 0:
        frame_ += 6
    else:
        frame_ += 10

    plt.show()



# 最后停留在中间
plt.pause(1)
plt.figure(figsize=(7.2, 5.76))

im_file = './imgs/' + str(test_id) + '/' + str(original_frame_) + '.jpg'
img = mpimg.imread(im_file)
plt.imshow(img)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)


plt.plot([], [], "r-", lw=3.0, label='Observed')
plt.plot([], [], "r--", lw=3.0, label='Ground truth')
plt.plot([], [], "g--", lw=3.0, label='SRAI-LSTM')
plt.plot([], [], "y--", lw=3.0, label='SR-LSTM')
# plt.plot([], [], "b--", lw=3.0, label='S-GAN')

for i in range(gx1.shape[-1]):
    # if i in [2, 3]:
    #     plt.plot(px4[-13:, i], py4[-13:, i], "b--", lw=1.5)

    if i in [4, 5]:
        gx_o, gx_p = gx1[:8, i], gx1[7:, i]
        gy_o, gy_p = gy1[:8, i], gy1[7:, i]
        plt.plot(gx_o, gy_o, "r-", lw=1.5)
        plt.plot(gx_p, gy_p, "r--", lw=1.5)

        plt.plot(px1[-13:, i], py1[-13:, i], "g--", lw=1.5)
        plt.plot(px2[-13:, i], py2[-13:, i], "y--", lw=1.5)

plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right')


# plt.savefig(save_path)
plt.show()

plt.ioff()
plt.close()

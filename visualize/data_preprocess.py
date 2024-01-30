# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:00:17 2019
data_preprocess.py
@author: AORUS
"""
from utils1 import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set()
# plt.ion()

H_dirs = ['./homography/eth.txt', './homography/hotel.txt',
          './homography/zara.txt', './homography/zara.txt',
          './homography/univ.txt']
models = ['sgan', 'srlstm', 'sralstm']

test_id = 2
model = models[2]
frame_ = 1680
H_ = torch.from_numpy(numpy.loadtxt(H_dirs[test_id]).astype(np.float32))
H = np.linalg.pinv(H_)

# frame_data的数据来自 batchs
'''
gt_path = './frame_data/' + model + '_' + str(test_id) + '_' + str(frame) + '_gt.npy'
pt_path = './frame_data/' + model + '_' + str(test_id) + '_' + str(frame) + '_pt.npy'
gt = np.load(gt_path)
pt = np.load(pt_path)

if test_id == 0:
    frame_ = frame + 42
else:
    frame_ = frame + 70
image_path = './images/' + str(test_id) + '_' + str(frame_) + '.jpg'
save_path = './save_images/' + str(test_id) + '_' + str(frame_) + '_' + model + '.png'

visualization(test_id, gt, pt, H, image_path, frame_, save_path)
'''

if test_id == 0:
    frame = frame_ - 42
else:
    frame = frame_ - 70


# id = 0
# if test_id == 4:
#     id = 1

gt_text1 = './batchs/' + str(test_id) + '/batch_sralstm/0_' + str(frame) + 'gt.npy'
pt_text1 = './batchs/' + str(test_id) + '/batch_sralstm/0_' + str(frame) + 'pt.npy'
gt_text2 = './batchs/' + str(test_id) + '/batch_srlstm/' + str(frame) + '_gt.npy'
pt_text2 = './batchs/' + str(test_id) + '/batch_srlstm/' + str(frame) + '_pt.npy'
gt_text3 = './batchs/' + str(test_id) + '/batch_srgat/' + str(frame) + '_g.npy'
pt_text3 = './batchs/' + str(test_id) + '/batch_srgat/' + str(frame) + '_p.npy'
gt_text4 = './batchs/' + str(test_id) + '/batch_sgan_m/' + str(frame) + '_g.npy'
pt_text4 = './batchs/' + str(test_id) + '/batch_sgan_m/' + str(frame) + '_p.npy'

gx1, gy1 = load2pixel(gt_text1, H)
px1, py1 = load2pixel(pt_text1, H)

gx2, gy2 = load2pixel(gt_text2, H, iftrans=True)
px2, py2 = load2pixel(pt_text2, H, iftrans=True)

gx3, gy3 = load2pixel(gt_text3, H)
px3, py3 = load2pixel(pt_text3, H)

gx4, gy4 = load2pixel(gt_text4, H)
px4, py4 = load2pixel(pt_text4, H)

# gx3, gy3, px3, py3 = gx3[0], gy3[0], px3[0], py3[0]
print(gx1.shape, gx2.shape, gx3.shape, gx4.shape)

plt.figure(figsize=(7.2, 5.76))

# im_file = './imgs/' + str(test_id) + '/' + str(frame_) + '.jpg'
# img = mpimg.imread(im_file)
# plt.imshow(img)
#
# plt.pause(0.1) #每0.4s暂停一下
# plt.clf() #清除控制台

im_file1 = './imgs/' + str(test_id) + '/' + str(frame) + '.jpg'
img1 = mpimg.imread(im_file1)
plt.imshow(img1)


plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)


if test_id == 0 or test_id == 1:
    gx1, gy1, px1, py1 = gy1, gx1, py1, px1
    gx2, gy2, px2, py2 = gy2, gx2, py2, px2
    gx3, gy3, px3, py3 = gy3, gx3, py3, px3

    # gy1, gy2, py1, py2 = gy1 + 50, gy2 + 50, py1 + 50, py2 + 50

# plt.plot([], [], "r--", lw=3.0, label='Ground truth')
# plt.plot([], [], "g--", lw=3.0, label='SRAI-LSTM')
# plt.plot([], [], "y--", lw=3.0, label='SR-LSTM')
#
# for i in range(gx1.shape[-1]):
#     if i in [2, 3]:
#         gx_o, gx_p = gx1[:8, i], gx1[8:, i]
#         gy_o, gy_p = gy1[:8, i], gy1[8:, i]
#         plt.plot(gx_o, gy_o, "r-", lw=3.0)
#         plt.plot(gx_p, gy_p, "r--", lw=3.0)
#         # 这一行
#         plt.plot(px1[-13:, i], py1[-13:, i], "g--", lw=3.0)
#         # ax = sns.kdeplot(gx_p, gy_p, cmap='Blues', shade=False, shade_lowest=True, n_levels=15)
#         plt.plot(px2[-13:, i], py2[-13:, i], "y--", lw=3.0)


# 多模态轨迹
gx3, gy3 = gx3[0], gy3[0]

for i in range(gx3.shape[-1]):
    if i in [1]:
        gx_o, gx_p = gx3[:9, i], gx3[8:, i]
        gy_o, gy_p = gy3[:9, i], gy3[8:, i]

        for j in range(1):
            # plt.plot(px3[:, j, i], py3[:, j, i], "r--", lw=3.0)
            plt.plot(px4[j, -13:, i], py4[j, -13:, i], "g--", lw=3.0)

        plt.plot(gx_p, gy_p, "b--", lw=3.0)


plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right')

# plt.title('Frame:' + str(frame_))
# plt.savefig(save_path)
plt.show()

# plt.ioff()

plt.close()

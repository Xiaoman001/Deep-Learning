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
sns.set()


H_dirs = ['./homography/eth.txt', './homography/hotel.txt',
          './homography/zara.txt', './homography/zara.txt',
          './homography/univ.txt']

test_id = 3

frame_ = 9760
H_ = torch.from_numpy(numpy.loadtxt(H_dirs[test_id]).astype(np.float32))
H = np.linalg.pinv(H_)


if test_id == 0:
    frame = frame_ - 42
else:
    frame = frame_ - 70


id = 0
# if test_id == 4:
#     id = 1

gt_text1 = './batchs/' + str(test_id) + '/batch_stirnet_m/' + str(frame) + '_gt.npy'
pt_text1 = './batchs/' + str(test_id) + '/batch_stirnet_m/' + str(frame) + '_pt.npy'
gx1, gy1 = load2pixel(gt_text1, H)
px, py = load2pixel(pt_text1, H)
print(gx1.shape, gy1.shape, px.shape, py.shape)
k = 4
px1, py1 = px[k], py[k]

# gt_text1 = './batchs/' + str(test_id) + '/batch_stgat/' + str(frame) + '_g.npy'
# pt_text1 = './batchs/' + str(test_id) + '/batch_stgat/' + str(frame) + '_p.npy'
# gx, gy = load2pixel(gt_text1, H, iftrans=False)
# px, py = load2pixel(pt_text1, H, iftrans=False)
# print(gx.shape, gy.shape, px.shape, py.shape)
# k = 0
# gx1, gy1 = gx[k], gy[k]
# px1, py1 = px[k], py[k]

# gt_text1 = './batchs/' + str(test_id) + '/batch_nmmp/' + str(frame) + '_g.npy'
# pt_text1 = './batchs/' + str(test_id) + '/batch_nmmp/' + str(frame) + '_p.npy'
# gx, gy = load2pixel(gt_text1, H, iftrans=False)
# px, py = load2pixel(pt_text1, H, iftrans=False)
# print(gx.shape, gy.shape, px.shape, py.shape)
# k = 0
# gx1, gy1 = gx[k], gy[k]
# px1, py1 = px[k], py[k]

# gx3, gy3, px3, py3 = gx3[0], gy3[0], px3[0], py3[0]

im_file = './imgs/' + str(test_id) + '/' + str(frame_) + '.jpg'
img = mpimg.imread(im_file)
im_file1 = './imgs/mask.png'
img = mpimg.imread(im_file)
img1 = mpimg.imread(im_file1)
plt.imshow(img)
plt.imshow(img1[:img.shape[0], :img.shape[1], :])
# print(gx1.shape)
if test_id == 0 or test_id == 1:
    gx1, gy1, px1, py1 = gy1, gx1, py1, px1

for i in range(gx1.shape[-1]):
    if i in [1, 2]:
        gx_o, gx_p = gx1[:9, i], gx1[8:, i]
        gy_o, gy_p = gy1[:9, i], gy1[8:, i]
        plt.plot(gx_o, gy_o, ls="-", markersize=4, lw=2.0, c="blue")
        plt.plot(gx_p, gy_p, ls="-", markersize=6, lw=2.0, c="yellow")
        plt.plot(px1[-12:, i], py1[-12:, i], marker=".", ls="-", markersize=6, lw=2.0, c="red")
        # ax = sns.kdeplot(gx_p, gy_p, cmap='Blues', shade=False, shade_lowest=True, n_levels=15)

# 多模态轨迹
# gx3, gy3 = gx3[0], gy3[0]
#
# for i in range(gx3.shape[-1]):
#     if i in [1]:
#         gx_o, gx_p = gx3[:9, i], gx3[8:, i]
#         gy_o, gy_p = gy3[:9, i], gy3[8:, i]
#         plt.plot(gx_o, gy_o, "b-", lw=4.0)
#         x = px3[:, 2:, i].reshape(-1)
#         y = py3[:, 2:, i].reshape(-1)
#         print(x.shape, y.shape)
#         for j in range(1):
#             # plt.plot(px3[:, j, i], py3[:, j, i], "r--", lw=1.0)
#             # plt.plot(px4[j, -13:, i], py4[j, -13:, i], "r--", lw=1.0)
#             ax = sns.kdeplot(x, y, shade=True, shade_lowest=False)
#
#         plt.plot(gx_p, gy_p, "b--", lw=2.0)


plt.xticks([])
plt.yticks([])
# 添加标题，显示绘制的图像
# plt.title('Frame:' + str(frame_))
# plt.savefig(save_path)
plt.show()

plt.close()

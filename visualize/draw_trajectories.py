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

test_id = 2

frame_ = 5930
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

gt_text1 = './batchs/' + str(test_id) + '/batch_srgat/' + str(frame) + '_g.npy'
pt_text1 = './batchs/' + str(test_id) + '/batch_stgat/' + str(frame) + '_p.npy'

print(pt_text1)
gx1, gy1 = load2pixel(gt_text1, H)
px1, py1 = load2pixel(pt_text1, H)
gx1, gy1 = gx1[1], gy1[1]
print(gx1.shape, gy1.shape, px1.shape, py1.shape)

im_file = './imgs/' + str(test_id) + '/' + str(frame_) + '.jpg'
# img = mpimg.imread(im_file)
im_file1 = './imgs/mask.png'
img = mpimg.imread(im_file)
img1 = mpimg.imread(im_file1)
plt.imshow(img)
plt.imshow(img1[:img.shape[0], :img.shape[1], :])

if test_id == 0 or test_id == 1:
    gx1, gy1, px1, py1 = gy1, gx1, py1, px1

# 多模态轨迹
# print(px1.shape)
i, j, k, w = 0, 0, 0, 1
gx_o, gx_p = gx1[:9, i], gx1[8:, i]
gy_o, gy_p = gy1[:9, i], gy1[8:, i]
plt.plot(gx_o, gy_o, ls="-", c="blue", lw=2.0)
# plt.plot(gx_p, gy_p, ls="--", c="blue", lw=2.0)
x = px1[:, -12:, i].reshape(-1)
y = py1[:, -12:, i].reshape(-1)
# print(gx_p[0], gy_p[0])
ax = sns.kdeplot(x=x, y=y, fill=False, n_levels=500, cmap="Blues")
# for t in range(20):
#     plt.plot(px1[t, 8, i], py1[t, 8, i], ls="--", c="blue", lw=2.0, marker="*")

gx_o, gx_p = gx1[:9, k], gx1[8:, k]
gy_o, gy_p = gy1[:9, k], gy1[8:, k]
plt.plot(gx_o, gy_o, ls="-", c="orange", lw=2.0)
# plt.plot(gx_p, gy_p, ls="--", c="orange", lw=2.0)
x = px1[:, -12:, k].reshape(-1)
y = py1[:, -12:, k].reshape(-1)
ax = sns.kdeplot(x=x, y=y, fill=False, n_levels=500, cmap="Oranges")
# for t in range(20):
#     plt.plot(px1[t, 7:9, k], py1[t, 7:9, k], ls="--", c="orange", lw=2.0)

gx_o, gx_p = gx1[:9, w], gx1[8:, w]
gy_o, gy_p = gy1[:9, w], gy1[8:, w]
plt.plot(gx_o, gy_o, ls="-", c="red", lw=2.0)
# plt.plot(gx_p, gy_p, ls="--", c="red", lw=2.0)
x = px1[:, -12:, w].reshape(-1)
y = py1[:, -12:, w].reshape(-1)
ax = sns.kdeplot(x=x, y=y, fill=False, n_levels=500, cmap="Reds")
# for t in range(20):
#     plt.plot(px1[t, 7:9, w], py1[t, 7:9, w], ls="--", c="red", lw=2.0)

gx_o, gx_p = gx1[:9, j], gx1[8:, j]
gy_o, gy_p = gy1[:9, j], gy1[8:, j]
plt.plot(gx_o, gy_o, ls="-", c="green", lw=2.0)
# plt.plot(gx_p, gy_p, ls="--", c="green", lw=2.0)
x = px1[:, -12:, j].reshape(-1)
y = py1[:, -12:, j].reshape(-1)
ax = sns.kdeplot(x=x, y=y, fill=False, n_levels=500, cmap="Greens")
# for t in range(20):
#     plt.plot(px1[t, 9, j], py1[t, 9, j], ls="--", c="green", lw=2.0, marker="*")

plt.xticks([])
plt.yticks([])
# 添加标题，显示绘制的图像
# plt.title('Frame:' + str(frame_))
# plt.savefig(save_path)
plt.show()

plt.close()

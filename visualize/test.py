import numpy as np
from utils import *
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import matplotlib.image as mpimg


H_dirs = ['./homography/eth.txt', './homography/hotel.txt',
          './homography/zara.txt', './homography/zara.txt',
          './homography/univ.txt']
H_ = np.loadtxt(H_dirs[0]).astype(np.float32)
H = np.linalg.inv(H_)
im_file = './eth.jpg'
img = mpimg.imread(im_file)
[height, weight, C] = img.shape
x0, y0 = weight //2, height //2
M = np.array([0, 0, 1])
P = np.array([x0, y0, 1])
# print(M.shape, P.shape)
# H_new = np.matmul(M.T, np.linalg.inv(P))
# H = np.linalg.inv(H_new)
T0 = np.array([0, height, 1])
T1 = np.array([weight, height, 1])
T2 = np.array([0, height//2, 1])
B0 = np.array([0, 0, 1])
B1 = np.array([weight, 0, 1])
C = np.array([weight//2, height//2, 1])
t0 = np.matmul(H_, T0.T)
t1 = np.matmul(H_, T1.T)
t2 = np.matmul(H_, T2.T)
b0 = np.matmul(H_, B0.T)
b1 = np.matmul(H_, B1.T)
c = np.matmul(H_, C.T)
print(t0, t1, b0, b1)
print(c)
plt.figure(0)
plt.plot([t0[0], t1[0], b1[0], b0[0], t0[0]], [t0[1], t1[1], b1[1], b0[1], t0[1]], marker=".", markersize=8, ls="-", lw=4)
plt.plot([c[0]], [c[1]], marker="*", markersize=16, color="red")
plt.plot([t2[0]], [t2[1]], marker="*", markersize=16, color="green")
plt.show()

plt.figure(1)
m1 = np.array([0, 0, 1])
m2 = np.array([0, 1, 1])
m3 = np.array([1, 0, 1])
p1 = np.matmul(H, m1.T)
# print(m1, m1.T)
p2 = np.matmul(H, m2.T)
p3 = np.matmul(H, m3.T)
# print(p1, p2, p3)
p11 = p1 / p1[2]
p22 = p2 / p2[2]
p33 = p3 / p3[2]
# print(p22[1]-p11[1], p33[0]-p11[0])
plt.imshow(img)
plt.plot(p11[0], p11[1], "r*")
plt.plot(p22[0], p22[1], "b*")
plt.plot(p33[0], p33[1], "g*")
plt.show()
'''
seg_map = np.load('./eth.npy')
im_file = './eth.jpg'
seg_img = seg2rgb(seg_map)
image = np.asanyarray(seg_img, dtype=np.uint8)
#image = np.fliplr(image)
#image = np.rot90(image, 1, (0, 1))

img = mpimg.imread(im_file)
print(img.shape)
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(img)
plt.show()
'''

from utils1 import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


H_dirs = ['./homography/eth.txt', './homography/hotel.txt',
          './homography/zara.txt', './homography/zara.txt',
          './homography/univ.txt']

pixel = [4763, 10550, 7850, 1320, 2180]

test_id = 0

frame_ = pixel[test_id]

H_ = torch.from_numpy(numpy.loadtxt(H_dirs[test_id]).astype(np.float32))
H = np.linalg.pinv(H_)

if test_id == 0:
    frame = frame_ - 42
else:
    frame = frame_ - 70
# print("H",H)

# 读取像素
gt_text1 = './batchs/' + str(test_id) + '/batch_sralstm/0_' + str(frame) + 'gt.npy'
pt_text1 = './batchs/' + str(test_id) + '/batch_sralstm/0_' + str(frame) + 'pt.npy'
gt_text2 = './batchs/' + str(test_id) + '/batch_srlstm/' + str(frame) + '_gt.npy'
pt_text2 = './batchs/' + str(test_id) + '/batch_srlstm/' + str(frame) + '_pt.npy'
gt_text4 = './batchs/' + str(test_id) + '/batch_sgan/sgan_' + str(frame) + '_gt.npy'
pt_text4 = './batchs/' + str(test_id) + '/batch_sgan/sgan_' + str(frame) + '_pt.npy'

# gx1, gy1 = load2pixel(gt_text1, H)
px1, py1 = load2pixel(pt_text1, H)

# print(gx1.shape)
# print(gx1.shape[-1])
# print(gx1, gy1)
# print(gx1.shape[0])

# for i in range (gx1.shape[0]):
#     print(gx1[i, 2],gy1[i, 2])
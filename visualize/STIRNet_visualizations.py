from utils1 import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


H_dirs = ['./homography/eth.txt', './homography/hotel.txt',
          './homography/zara.txt', './homography/zara.txt',
          './homography/univ.txt']

test_id = 2

frame_ = 1640
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

gt_text2 = './batchs/' + str(test_id) + '/batch_stgat/' + str(frame) + '_g.npy'
pt_text2 = './batchs/' + str(test_id) + '/batch_stgat/' + str(frame) + '_p.npy'
# print(pt_text1)
gx1, gy1 = load2pixel(gt_text1, H)
px1, py1 = load2pixel(pt_text1, H)

print('维度：', gx1.shape, gy1.shape, px1.shape, py1.shape)
gx2, gy2 = load2pixel(gt_text2, H)
px2, py2 = load2pixel(pt_text2, H)
# print(gx2.shape, gy2.shape, px2.shape, py2.shape)
im_file = './imgs/' + str(test_id) + '/' + str(frame_) + '.jpg'
img = mpimg.imread(im_file)
im_file1 = './imgs/mask1.png'
img = mpimg.imread(im_file)
img1 = mpimg.imread(im_file1)
plt.imshow(img)
plt.imshow(img1[:img.shape[0], :img.shape[1], :])

if test_id == 0 or test_id == 1:
    gx1, gy1, px1, py1 = gy1, gx1, py1, px1
    gx2, gy2, px2, py2 = gy2, gx2, py2, px2

# 多模态轨迹
gx, gy = gx2[0], gy2[0]
i, j, k, w = 1, 2, 3, 4
gx_o, gy_o, gx_p, gy_p = gx[:8, i], gy[:8, i], gx[-12:, i], gy[-12:, i]
plt.plot(gx_o, gy_o, ls="-", c="blue", lw=2.0)
plt.scatter(gx_p, gy_p, marker=".", c="blue", s=25.0)
gx_o, gy_o, gx_p, gy_p = gx[:8, j], gy[:8, j], gx[-12:, j], gy[-12:, j]
plt.plot(gx_o, gy_o, ls="-", c="green", lw=2.0)
plt.scatter(gx_p, gy_p, marker=".", c="green", s=25.0)
gx_o, gy_o, gx_p, gy_p = gx[:8, k], gy[:8, k], gx[-12:, k], gy[-12:, k]
plt.plot(gx_o, gy_o, ls="-", c="red", lw=2.0)
plt.scatter(gx_p, gy_p, marker=".", c="red", s=25.0)
gx_o, gy_o, gx_p, gy_p = gx[:8, w], gy[:8, w], gx[-12:, w], gy[-12:, w]
plt.plot(gx_o, gy_o, ls="-", c="purple", lw=2.0)
plt.scatter(gx_p, gy_p, marker=".", c="purple", s=25.0)
plt.xticks([])
plt.yticks([])
save_path = './MTP_cases/zara1_' + str(frame_) + '_stirner_gt.png'
plt.savefig(save_path)
plt.show()
print(img1.shape)
for num in range(20):
    # plt.imshow(img)
    plt.imshow(img1[:img.shape[0], :img.shape[1], :])
    # gx, gy = gx2[0], gy2[0]  # STGAT
    # gx, gy = gx1, gy1  # STIRNet
    px, py = px1[num], py1[num]  # STIRNet
    # px, py = px2[num], py2[num]  # STGAT
    print(gx.shape, px.shape, px.shape)
    # i, j, k, w = 1, 2, 3, 4
    gx_o, gy_o, px_p, py_p = gx[:8, i], gy[:8, i], px[-12:, i], py[-12:, i]
    plt.plot(gx_o, gy_o, ls="-", c="blue", lw=2.0)
    plt.scatter(px_p, py_p, marker=".", c="blue", s=25.0)

    gx_o, gy_o, px_p, py_p = gx[:8, j], gy[:8, j], px[-12:, j], py[-12:, j]
    plt.plot(gx_o, gy_o, ls="-", c="green", lw=2.0)
    plt.scatter(px_p, py_p, marker=".", c="green", s=25.0)

    gx_o, gy_o, px_p, py_p = gx[:8, k], gy[:8, k], px[-12:, k], py[-12:, k]
    plt.plot(gx_o, gy_o, ls="-", c="red", lw=2.0)
    plt.scatter(px_p, py_p, marker=".", c="red", s=25.0)

    gx_o, gy_o, px_p, py_p = gx[:8, w], gy[:8, w], px[-12:, w], py[-12:, w]
    plt.plot(gx_o, gy_o, ls="-", c="purple", lw=2.0)
    plt.scatter(px_p, py_p, marker=".", c="purple", s=25.0)

    plt.xticks([])
    plt.yticks([])
    # 添加标题，显示绘制的图像
    save_path = './MTP_cases/zara1_' + str(frame_) + '_stirner_pt' + str(num+1) + '.png'
    # save_path = './MTP_cases/zara1_' + str(frame_) + '_stgat_pt' + str(num+1) + '.png'
    plt.savefig(save_path)
    plt.show()
    # plt.pause(1)

plt.close()

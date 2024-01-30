# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:00:17 2019
data_preprocess.py
@author: AORUS
"""
from utils1 import *
from PIL import Image
import matplotlib.pyplot as plt

test_id = 4
data_dirs = ['./data/eth/univ', './data/eth/hotel', './data/ucy/zara/zara01',
             './data/ucy/zara/zara02', 'data/ucy/univ/students003',
             './data/ucy/univ/students001', './data/ucy/univ/uni_examples',
             './data/ucy/zara/zara03']
video_dirs = ['./datasets/video/eth.avi', './datasets/video/hotel.avi',
              './datasets/video/zara01.avi', './datasets/video/zara02.avi',
              './datasets/video/univ.avi']
H_dirs = ['./datasets/homography/eth.txt', './datasets/homography/hotel.txt',
          './datasets/homography/zara.txt', './datasets/homography/zara.txt',
          './datasets/homography/univ.txt']

filename = data_dirs[test_id]
video_path = video_dirs[test_id]
image_path = './datasets/' + str(test_id) + '/images/'
# prepare(filename, video_path, image_path)
'''
frame = 780
gt_path = './savedata/' + str(test_id) + '/sralstm/batch/' + '0_' + \
          str(frame) + 'gt.npy'
pt_path = './savedata/' + str(test_id) + '/sralstm/batch/' + '0_' + \
          str(frame) + 'pt.npy'
gt = np.load(gt_path)
pt = np.load(pt_path)
if test_id == 0:
    frame = frame + 48
else:
    frame = frame + 80

im_file = image_path + str(frame) + '.jpg'
H = numpy.loadtxt(H_dirs[test_id]).astype(np.float32)
H_t = torch.pinverse(torch.from_numpy(H))

save_path = './datasets/' + str(test_id) + '/save_images/'
# visualization(gt, pt, H_t, im_file, frame, save_path)
path1 = os.path.join(data_dirs[0], 'true_pos_.csv')
path2 = os.path.join(data_dirs[0], 'biwi_eth.txt')
csv2txt(path1, path2)
'''
result = np.array([0, 0, 0, 0, 0, 0, 0])
num = 0
for i in range(5):
    final_dir = './savedata/' + str(i) + '/sralstm/final_error/'
    all_files = os.listdir(final_dir)
    all_files = [os.path.join(final_dir, _path) for _path in all_files]
    final = []
    for file in all_files:
        fn = np.load(file)
        final.extend(fn)
    print(i, final.__len__())
    hist = np.histogram(final, bins=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 10])
    result = result + np.array(hist[0])
    num += final.__len__()
print(result/num)
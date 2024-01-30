import numpy as np


def seg2rgb(seg_map):

    [h, w, d] = seg_map.shape
    print(seg_map.shape)
    color = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
                    [255, 0, 255], [255, 97, 0], [30, 144, 255]])
    seg_img = np.zeros([h, w, 3])
    seg_map_ = seg_map.reshape(h*w, d)
    seg_img_ = seg_img.reshape(h*w, 3)
    for i in range(8):
        ind = np.where(seg_map_[:, i] > 0)[0]
        seg_img_[ind] = color[i]

    seg_img = seg_img_.reshape(h, w, 3)
    return seg_img
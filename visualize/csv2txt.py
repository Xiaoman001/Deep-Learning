import numpy as np


data = np.genfromtxt("./eth.csv", delimiter=',')
data = data.transpose(0, 1)
print(data.shape)
np.savetxt("./biwi_eth.txt", data)
# fsave = open("./eth.txt", mode="a")
# for i in range(data.shape[1]):
#     fsave.write(int(data[0, i]) + " " + int(data[1, i]) + " " + round(data[2, i], 4) + " " + round(data[2, i], 4) + "\n")
# fsave.close()


import cv2

def extract_frames(video_path, save_path, interval):
    vc = cv2.VideoCapture(video_path)
    frame_rate = vc.get(cv2.CAP_PROP_FPS)  # 视频帧率
    interval_frames = int(frame_rate * interval)  # 计算间隔的帧数
    current_frame = 0  # 当前帧索引
    saved_frame = 0  # 保存的帧计数

    while True:
        success, frame = vc.read()
        if not success:
            break

        if current_frame % interval_frames == 0:
            save_name = save_path + str(saved_frame * 10) + '.jpg'
            cv2.imwrite(save_name, frame)
            saved_frame += 1

        current_frame += 1

    vc.release()


video_path = 'path/to/video.mp4'
save_path = 'path/to/save/'
interval = 0.4  # 间隔时间，单位为秒

extract_frames(video_path, save_path, interval)

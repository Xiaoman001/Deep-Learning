import cv2
import numpy as np


cap = cv2.VideoCapture('./ETH/seq_eth/seq_eth.avi')
filename = './ETH/seq_eth/obsmat.txt'

while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
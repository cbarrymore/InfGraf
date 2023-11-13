import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt


# path = "C:/Users/cbarr/Downloads/USMNT vs. Ghana _ Highlights_ October 17, 2023.mp4"

# output = "detection_video.mp4"
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model.cuda()

img = cv2.imread(
    "10748_jpg.rf.c86bc9a4116f0102ad8ccde07a050dec.jpg", cv2.IMREAD_GRAYSCALE)

assert img is not None, "file could not be read, check with os.path.exists()"
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
print(hist)

plt.plot(hist)
plt.show()
# detection = model(img)
# print(detection[0])

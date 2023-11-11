import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt


# path = "C:/Users/cbarr/Downloads/USMNT vs. Ghana _ Highlights_ October 17, 2023.mp4"

# output = "detection_video.mp4"
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model.cuda()

img = cv2.imread("151_jpg.rf.c4c44e9a3e29493053f107c6ec46f7c2.jpg", cv2.IMREAD_GRAYSCALE)
img.shape[:]
assert img is not None, "file could not be read, check with os.path.exists()"
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
print(hist)

plt.plot(hist)
plt.show()
# detection = model(img)
# print(detection[0])

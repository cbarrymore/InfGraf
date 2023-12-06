import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from PIL import Image


K = 8
trackbar_changed = False


def color_quantize_fast(image, K):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(np.uint8(img))
    im_pil = im_pil.quantize(K, None, 0, None)
    return cv2.cvtColor(np.array(im_pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def on_trackbar_change(value):
    global K
    global trackbar_changed
    K = value
    trackbar_changed = True
    print(K)


cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 400, 80)

cv2.createTrackbar('K', 'Trackbars', K, 20, on_trackbar_change)

cap = cv2.VideoCapture('USMNT_vs._Ghana___Highlights__October_17_2023.mp4')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    Z = frame.reshape((-1, 3))
    Z = np.float32(Z)

    clt = MiniBatchKMeans(n_clusters=K, max_iter=10, n_init=1)
    clt.fit(Z)
    label = clt.labels_
    center = clt.cluster_centers_

    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    trackbar_changed = False
    res2 = color_quantize_fast(frame, K)
    cv2.imshow('res2', res2)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

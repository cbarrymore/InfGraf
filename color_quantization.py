import numpy as np
import cv2

K = 8
trackbar_changed = False


def on_trackbar_change(value):
    global K
    global trackbar_changed
    K = value
    trackbar_changed = True
    print(K)


cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', 400, 80)

cv2.createTrackbar('K', 'Trackbars', K, 20, on_trackbar_change)

img = cv2.imread(
    'bb1a9c9f-a09e-42d6-8fe9-8f9461790878_16-9-discover-aspect-ratio_default_0.jpg')
Z = img.reshape((-1, 3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

while True:
    if trackbar_changed:
        ret, label, center = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        print(center)
        print(label)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        cv2.imshow('res2', res2)
        trackbar_changed = False

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

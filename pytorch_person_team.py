import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

color = ('b', 'g', 'r')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.cuda()

img = cv2.imread(
    '10748_jpg.rf.c86bc9a4116f0102ad8ccde07a050dec.jpg')

mask = np.zeros_like(img)

mask = np.zeros(img.shape[:2], np.uint8)


result = model(img)
# cv2.imshow("a", img)
# cv2.waitKey(0)
result.print()
pred = result.xyxy[0]  # img1 predictions (tensor)

for det in pred:
    x_min, y_min, x_max, y_max, conf, class_id = det
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

    # Obtenemos el centro del rectangulo de detección
    x_center_of_rectangle, y_center_of_rectangle = int(
        (x_max+x_min)//2), int((y_max+y_min)//2)

    mask = np.zeros(img.shape[:2], np.uint8)

    detection_rectangle = img[y_min:y_max, x_min:x_max]

    # Sacamos la anchura y altura que queremos que tenga
    # la imagen del centro del rectangulo de deteccion
    center_width_min, center_width_max = (x_center_of_rectangle +
                                          x_min)//2, (x_center_of_rectangle+x_max)//2
    center_height_min, center_height_max = (y_center_of_rectangle +
                                            y_min)//2, (y_center_of_rectangle)

    # Sacamos de la imagen original la imagen del centro del rectangulo de deteccion
    detection_center_image = img[center_height_min:center_height_max,
                                 center_width_min:center_width_max]
    cv2.imshow("img", detection_center_image)
    cv2.imshow("img2", detection_rectangle)
    cv2.waitKey(0)
    # Sacamos las dimensiones del centro de deteccion
    height, width, _ = np.shape(detection_center_image)

    # Hacemos clustering
    data = np.reshape(detection_center_image, (height * width, 3))
    data = np.float32(data)

    number_clusters = 1

    compactness, labels, centers = cv2.kmeans(
        data, number_clusters, None, criteria, 10, flags)

    print(centers)
    for index, row in enumerate(centers):
        blue, green, red = int(row[0]), int(row[1]), int(row[2])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                      (blue, green, red), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)
quit()

mask[center_height_min:center_height_max,
     center_width_min:center_width_max] = 255

masked_img = cv2.bitwise_and(img, img, mask=mask)


mask = np.zeros(img.shape[:2], np.uint8)

# Metodo 1 para hacer mascara
mask[y_min:y_max, x_min:x_max] = 255
cv2.imshow("Mask", mask)
cv2.waitKey(0)


masked_img = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Mask", masked_img)
cv2.waitKey(0)


masked_img_only = img[y_min:y_max, x_min:x_max]
cv2.imwrite("deteccion.jpg", masked_img_only)
height, width = y_max-y_min, x_max-x_min

height, width, _ = np.shape(img)

data = np.reshape(img, (height * width, 3))
data = np.float32(data)

number_clusters = 5

compactness, labels, centers = cv2.kmeans(
    data, number_clusters, None, criteria, 10, flags)

print(labels)
num_labels = np.arange(0, len(np.unique(labels)) + 1)
print(num_labels)
(hist, _) = np.histogram(labels, bins=num_labels)
hist = hist.astype("float")
hist /= hist.sum()

print(hist)
print(centers)
quit()
cv2.imshow("Mask", masked_img_only)
cv2.waitKey(0)

np.reshape

º
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])


h = np.zeros((300, 256, 3))
cv2.normalize(hist_mask, hist_mask, 0, 255, cv2.NORM_MINMAX)
hist = np.int32(np.around(hist_mask))
for x, y in enumerate(hist):
    cv2.line(h, (x, 0), (x, y[0]), (255, 255, 255))
y = np.flipud(h)


cv2.imshow("histograma", y)
cv2.imshow('image', masked_img)
cv2.waitKey(0)
print(hist_mask)
print(hist_full)
plt.plot(hist_full)
plt.show()

# #Metodo 2 para hacer máscara
# mask = np.zeros_like(img)

# # Dibujar un rectángulo blanco en la máscara
# cv2.rectangle(mask, (x_min, y_min), (x_max, y_max),
#               (255, 255, 255), thickness=cv2.FILLED)

# # Mostrar la imagen de la máscara
# cv2.imshow("Mask", mask)
# cv2.waitKey(0)
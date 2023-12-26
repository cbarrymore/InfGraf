import cv2
import numpy as np
import matplotlib.pyplot as plt
K = 8


def on_trackbar_change(value):
    global K
    global trackbar_changed
    K = value
    trackbar_changed = True


def find_closest_palette_color(pixel):
    distances = np.linalg.norm(color_palette - pixel, axis=1)
    index = np.argmin(distances)
    
    return color_palette[index]



def floyd_steinberg_dithering(image):
    print(image.shape)
    h, w = image.shape[:2]
    image = image.astype(int)
    for y in range(h):
        for x in range(w):
            old_pixel = image[y, x].copy()
            new_pixel = find_closest_palette_color(old_pixel)
            image[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < w:
                image[y, x + 1] = image[y, x + 1] + error * 7 / 16
            if y + 1 < h and x - 1 > 0:
                image[y + 1, x - 1] = image[y + 1, x - 1] + error * 3 / 16
            if y + 1 < h:
                image[y + 1, x] = image[y + 1, x] + error * 5 / 16
            if y + 1 < h and x + 1 < w:
                image[y + 1, x + 1] = image[y + 1, x + 1] + error * 1 / 16
    return image.astype(np.uint8)


def quantization_kmeans(Z, K, criteria, image):
    global color_palette
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    color_palette = center.copy()
    color_palette = color_palette.astype(int)
    res = center[label.flatten()]
    return res.reshape((image.shape)), color_palette


def create_color_palette_bar():
    height, width = 200, 200
    bars = []
    bgr_values = []
    for index, color in enumerate(color_palette):
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        red, green, blue = int(color[2]), int(color[1]), int(color[0])

        bgr = (blue, green, red)
        bars.append(bar)
        bgr_values.append(bgr)

        img_bar = np.hstack(bars)
        for index, row in enumerate(bgr_values):
            cv2.putText(img_bar, f'{index + 1}. RGB: {row}', (5 + 200 * index, 200 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img_bar


def quantization(self, image):
    global trackbar_changed
    
    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 80)

    cv2.createTrackbar('K', 'Trackbars', K, 64, on_trackbar_change)

    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    while True:
        if trackbar_changed and K != 0:

            k_means_image, color_palette = quantization_kmeans(
                Z, K, criteria, image)
            img_bar = create_color_palette_bar()
            if self.dithering == True:
                fs_dither_image = floyd_steinberg_dithering(image.copy())
                cv2.imshow('con dithering', fs_dither_image)

            quantization_image = k_means_image
            cv2.imshow('cuantificación sin dithering', quantization_image)
            trackbar_changed = False

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

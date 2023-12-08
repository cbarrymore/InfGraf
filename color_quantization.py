import numpy as np
import cv2
import matplotlib.pyplot as plt

K = 8
trackbar_changed = False
color_palette = None


def on_trackbar_change(value):
    global K
    global trackbar_changed
    K = value
    trackbar_changed = True
    print(K)


def find_closest_palette_color(pixel):
    # min euclidean distance between pixel and color_palette
    # return color_palette[index]
    distances = np.linalg.norm(color_palette - pixel, axis=1)
    index = np.argmin(distances)
    return color_palette[index]

    # Return the closest palette color


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


def quantization_kmeans(Z, K, criteria):
    global color_palette
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    for i in range(K):
        cluster_points = Z[label.flatten() == i]
        plt.scatter(cluster_points[:, 0],
                    cluster_points[:, 1], label=f'Cluster {i + 1}')

    # Plot the cluster centers
    plt.scatter(center[:, 0], center[:, 1], marker='X',
                s=200, c='red', label='Cluster Centers')

    plt.title('K-Means Clustering Results')
    plt.legend()
    plt.show()

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    color_palette = center.copy()
    # convert color_palette to int
    color_palette = color_palette.astype(int)
    res = center[label.flatten()]
    return res.reshape((img.shape))


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


cv2.namedWindow('Trackbars')

cv2.resizeWindow('Trackbars', 400, 80)

cv2.createTrackbar('K', 'Trackbars', K, 64, on_trackbar_change)

img = cv2.imread(
    'Soccer-EURO-2020-770x513.jpg')
Z = img.reshape((-1, 3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

while True:
    if trackbar_changed:

        k_means_image = quantization_kmeans(Z, K, criteria)
        img_bar = create_color_palette_bar()
        fs_dither_image = floyd_steinberg_dithering(img.copy())

        cv2.imshow('fs_dither_image', fs_dither_image)
        cv2.imshow('img_bar', img_bar)
        cv2.imshow('k_means_image', k_means_image)

        trackbar_changed = False

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

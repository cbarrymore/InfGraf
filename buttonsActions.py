from PySide2.QtCore import QCoreApplication, QRect, QMetaObject, Qt, QUrl, QTimer
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QMenuBar, QToolBar, QStatusBar, QFileDialog, QLabel
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
import cv2
import pytorch_media_detect
import torch
import numpy as np
from videoWindow import VideoWindow
import time


class ArchivoInfo:
    def __init__(self):
        self.nombre = ""
        self.tipo = ""


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
font = cv2.FONT_HERSHEY_SIMPLEX

archivo = ArchivoInfo()
count_frame = 0
color_count = {}
color_search = np.zeros((200, 200, 3), np.uint8)
color_selected = np.zeros((200, 200, 3), np.uint8)
hue = 0
saturation = 0
value = 0
mouse_callback_triggered = False
white_flag = False
black_flag = False

hue = 0
_y = 0
u = 0
v = 0
u_v_positive = False
u_v_negative = False
u_positive_v_negative = False
u_negative_v_positive = False
near_center = False
white = False
K = 8
trackbar_changed = False
cargada = False
ultimo_frame = None
video_window = None


def action_cargar_imagen(self):
    filename, _ = QFileDialog.getOpenFileName(
        self, "Seleccionar imagen", ".", "Images (*.png *.jpg *.bmp)")
    if filename:
        pixmap = QPixmap(filename)
        crear_nuevo_archivo(str(filename))


def action_cargar_video(self):
    filtro_video = "Archivos de video (*.avi *.mp4 *.mkv *.mov);;Todos los formatos de video (*.avi *.mp4 *.mkv *.mov);;Otros (*.*)"
    nombre, _ = QFileDialog.getOpenFileName(
        self, "Abrir video", ".", filtro_video)
    if nombre:
        crear_nuevo_archivo(str(nombre))
        self.label_2.setText(nombre)


def action_mostrar_contenido(self):
    global count_frame
    global cargada
    global ultimo_frame
    global video_window

    if video_window == None or not video_window.isVisible():
        video_window = VideoWindow()
        video_window.show()

    if archivo.tipo == "video":
        cap = cv2.VideoCapture(archivo.nombre)

        ret, frame = cap.read()

        # Crea una ventana con un nombre
        # cv2.namedWindow("Video")

        count_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            ultimo_frame = frame.copy()
            if not ret:
                break
            if self.object_detection == True:
                # call a function to detect objects
                frame = detect_objects(self, frame)
            if self.counting_HSV == True:
                do_image(frame)
                self.counting_HSV = False
            if self.counting_YUV == True:
                do_image_YUV(frame)
                self.counting_YUV = False
            if self.quantization == True:
                quantization(self, frame)
                self.quantization = False
            video_window.cargar_frame(frame)
            if not video_window.isVisible():
                break

            count_frame += 1
            if cv2.waitKey(1) > 0:
                break
        cap.release()
    else:
        img = cv2.imread(archivo.nombre)
        if self.counting_HSV == True:
            do_image(img)
        if self.counting_YUV == True:
            do_image_YUV(img)
        if self.quantization == True:
            quantization(self, img)
        if not self.counting_HSV and not self.counting_YUV and not self.quantization:
            if self.object_detection == True:
                # call a function to detect objects
                img = detect_objects(self, img)
            video_window.cargar_frame(img)


def crear_nuevo_archivo(nombre):
    archivo.nombre = nombre
    if archivo.nombre.split('.')[-1].lower() in ["jpg", "png", "bmp"]:
        archivo.tipo = "imagen"
    else:
        archivo.tipo = "video"


def action_activate_color_counting(self):
    if self.color_counting == True:
        self.color_counting = False
    else:
        self.color_counting = True
    if archivo.tipo == "imagen":
        self.mostrar_contenido()


def action_activar_object_detection(self):
    if self.object_detection == True:
        self.object_detection = False
    else:
        self.object_detection = True
    if archivo.tipo == "imagen":
        self.mostrar_contenido()


def action_activate_counting_HSV(self):
    if self.counting_HSV == True:
        self.counting_HSV = False
    else:
        self.counting_HSV = True
        self.counting_YUV = False
        self.quantization = False
    if archivo.tipo == "imagen":
        self.mostrar_contenido()


def action_activate_counting_YUV(self):
    if self.counting_YUV == True:
        self.counting_YUV = False
    else:
        self.counting_YUV = True
        self.counting_HSV = False
        self.quantization = False
    if archivo.tipo == "imagen":
        self.mostrar_contenido()


def action_activate_quantization(self):
    if self.quantization == True:
        self.quantization = False
    else:
        self.quantization = True
        self.counting_HSV = False
        self.counting_YUV = False
    if archivo.tipo == "imagen":
        self.mostrar_contenido()


def action_activate_dithering(self):
    if self.dithering == True:
        self.dithering = False
    else:
        self.dithering = True


def action_pause(self):
    global cargada

    if self.pause == True:
        self.pause = False
    else:
        self.pause = True
        cargada = False
    if archivo.tipo == "imagen":
        self.mostrar_contenido()


def detect_objects(self, frame):
    # Call the model to detect objects in the frame
    detection = self.model(frame)
    pred = detection.xyxy[0]  # img1 predictions (tensor)

    if self.object_detection:
        detection_frame = detection.render()[0]
    if self.color_counting == True:
        detection_frame = func_color_counting(pred, frame)
    return detection_frame


def distancia_color(color1, color2):
    return sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)) ** 0.5


def func_color_counting(pred, frame):
    global color_count
    global count_frame

    pred = pred[pred[:, 5] == 0]
    margen = 100
    if count_frame % 5 == 0:
        color_count = {}
    for det in pred:
        xmin, ymin, xmax, ymax, conf, class_id = det
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        # Obtenemos el centro del rectangulo de detección
        x_center_of_rectangle, y_center_of_rectangle = int(
            (xmax+xmin)//2), int((ymax+ymin)//2)
        # Sacamos la anchura y altura que queremos que tenga
        # la imagen del centro del rectangulo de deteccion
        center_width_min, center_width_max = (x_center_of_rectangle +
                                              xmin)//2, (x_center_of_rectangle+xmax)//2
        center_height_min, center_height_max = (y_center_of_rectangle +
                                                ymin)//2, (y_center_of_rectangle)
        # Sacamos de la imagen original la imagen del centro del rectangulo de deteccion
        detection_center_image = frame[center_height_min:center_height_max,
                                       center_width_min:center_width_max]
        # Sacamos las dimensiones del centro de deteccion
        height, width, _ = np.shape(detection_center_image)
        # Hacemos clustering
        data = np.reshape(detection_center_image, (height * width, 3))
        data = np.float32(data)
        median = np.mean(data, axis=0)
        # print("Median: ", median)
        number_clusters = 1
        compactness, labels, centers = cv2.kmeans(
            data, number_clusters, None, criteria, 10, flags)
        # print(centers)
        # for index, row in enumerate(centers):
        blue, green, red = int(median[0]), int(median[1]), int(median[2])

        color_encontrado = None
        distancia_minima = float('inf')
        if count_frame % 5 == 0:
            for color_key in color_count:
                # distancia_actual = distancia_color(color_key, (blue, green, red))
                distancia_actual = distancia_color(
                    color_key, (blue, green, red))
                if int(distancia_actual) <= margen and distancia_actual < distancia_minima:
                    color_encontrado = color_key
                    distancia_minima = distancia_actual

            if color_encontrado is not None:
                # Incrementa el contador para el color encontrado
                color_count[color_encontrado] += 1
            else:
                # Agrega una nueva entrada al diccionario si no se encontró un color similar
                color_count[(blue, green, red)] = 1
                color_encontrado = (blue, green, red)
        print("dibujamos rectangulo de color: ", blue, green, red)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                      (blue, green, red), 2)
        i = 0

    for color, count in color_count.items():
        cv2.putText(frame, f"Numero total: {count}", (0 + i, 0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color),
                    1)
        i += 150

    return frame

################### Formato HSV ################################


def select_color(event, x, y, flags, param):
    global hue
    global saturation
    global value
    global white_flag
    global black_flag

    B = frame[y, x][0]
    G = frame[y, x][1]
    R = frame[y, x][2]
    color_search[:] = (B, G, R)

    if event == cv2.EVENT_LBUTTONDOWN:
        white_flag = False
        black_flag = False
        color_selected[:] = (B, G, R)
        hue = hsv[y, x][0]
        saturation = hsv[y, x][1]
        value = hsv[y, x][2]
        if saturation <= 40 and value >= 100:
            white_flag = True
        elif value <= 40:
            black_flag = True

        print(hsv[y, x])
    mouse_callback_triggered = True  # Se activa la variable global


def search_contours(mask):
    contours_count = 0
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            contours_count += 1
    return contours_count


def nothing(x):
    pass


def do_image(image):
    global frame
    global hsv
    global video_window

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_color)

    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 80)

    cv2.createTrackbar('Lower-Hue', 'Trackbars', 14, 179, nothing)
    cv2.createTrackbar('Upper-Hue', 'Trackbars', 18, 179, nothing)

    while True:
        frame = image.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        diff_lower_hue = cv2.getTrackbarPos('Lower-Hue', 'Trackbars')
        diff_upper_hue = cv2.getTrackbarPos('Upper-Hue', 'Trackbars')

        lower_hue = 0 if hue - diff_lower_hue < 0 else hue - diff_lower_hue
        upper_hue = hue + diff_upper_hue if hue + diff_upper_hue < 179 else 179

        lower_saturation = 50
        upper_saturation = 255
        lower_value = 20
        upper_value = 255

        if white_flag:
            lower_saturation = 0
            upper_saturation = 40
            lower_value = 100
            upper_value = 255
        elif black_flag:
            lower_saturation = 0
            upper_saturation = 255
            lower_value = 0
            upper_value = 40
        if white_flag or black_flag:
            lower_hue = 0
            upper_hue = 179
        lower_hsv = np.array([lower_hue, lower_saturation, lower_value])
        upper_hsv = np.array([upper_hue, upper_saturation, upper_value])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        count = search_contours(mask)

        cv2.putText(frame, f'Total: {count}', (5, 30),
                    font, 1, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('mask', mask)
        cv2.imshow('image', frame)
        video_window.cargar_frame(frame)
        # cv2.imshow('image', frame)
        cv2.imshow('color_search', color_search)
        cv2.imshow('color_selected', color_selected)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


################### Formato YUV ################################


def select_color_YUV(event, x, y, flags, param):
    global hue
    global _y
    global u
    global mouse_callback_triggered
    global u_v_positive
    global u_v_negative
    global u_positive_v_negative
    global u_negative_v_positive
    global near_center
    global white
    global v
    B = frame[y, x][0]
    G = frame[y, x][1]
    R = frame[y, x][2]
    color_search[:] = (B, G, R)

    if event == cv2.EVENT_LBUTTONDOWN:
        color_selected[:] = (B, G, R)
        _y = yuv[y, x][0]
        u = yuv[y, x][1]
        v = yuv[y, x][2]
        # check if u and v are close to 127 by a margin of 10
        if (_y >= 127 or _y < 127) and abs(u - 127) <= 10 and abs(v - 127) <= 10:
            near_center = True
            # if _y -64 < 127 by a margin of 10
            white = _y > 127

        if u > 127 and v > 127:
            u_v_positive = True
        elif u < 127 and v > 127:
            u_negative_v_positive = True
        elif u > 127 and v < 127:
            u_positive_v_negative = True
        else:
            u_v_negative = True
        print(yuv[y, x])
        mouse_callback_triggered = True  # Se activa la variable global


def search_contours_YUV(mask):
    contours_count = 0
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            contours_count += 1

    return contours_count


def filter_colors_YUV(yuv_image, radius):
    global u
    global v
    distances = np.sqrt(((yuv_image[:, :, 0] - _y) ** 2 + yuv_image[:, :, 1] - u) ** 2 +
                        (yuv_image[:, :, 2] - v) ** 2)

    # Crear una máscara para los píxeles dentro del radio especificado
    mask = distances <= radius

    return mask


def do_image_YUV(image):
    global frame
    global yuv
    global mouse_callback_triggered
    global near_center
    global white
    global u_v_positive
    global u_v_negative
    global u_positive_v_negative
    global u_negative_v_positive
    global video_window

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_color_YUV)

    frame = image.copy()
    mask = None
    while True:
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        if mouse_callback_triggered:
            frame = image.copy()
            if near_center:
                if white:
                    print("white")
                    mask = np.logical_and(
                        yuv[:, :, 0] > 127, np.logical_and(abs(yuv[:, :, 1] - 127) <= 10, abs(yuv[:, :, 2] - 127) <= 10))
                    # if abs(u - 127) <= 10 and abs(v - 127) <= 10:
                else:
                    print("black")
                    mask = np.logical_and(
                        yuv[:, :, 0] < 127, np.logical_and(abs(yuv[:, :, 1] - 127) <= 20, abs(yuv[:, :, 2] - 127) <= 20))
            else:
                if u_v_positive:
                    mask = np.logical_and(
                        yuv[:, :, 1] > 127, yuv[:, :, 2] > 127)
                elif u_v_negative:
                    mask = np.logical_and(
                        yuv[:, :, 1] < 127, yuv[:, :, 2] < 127)
                elif u_positive_v_negative:
                    mask = np.logical_and(
                        yuv[:, :, 1] > 127, yuv[:, :, 2] < 127)
                else:
                    mask = np.logical_and(
                        yuv[:, :, 1] < 127, yuv[:, :, 2] > 127)
            mouse_callback_triggered = False
            white = False
            near_center = False
            u_v_negative = False
            u_v_positive = False
            u_positive_v_negative = False
            u_negative_v_positive = False
            mask = mask.astype(np.uint8) * 255
            count = search_contours_YUV(mask)
            cv2.putText(frame, f'Total: {count}', (5, 30),
                        font, 1, (255, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('mask', mask)
        video_window.cargar_frame(frame)
        cv2.imshow('image', frame)
        cv2.imshow('color_search', color_search)
        cv2.imshow('color_selected', color_selected)
        cv2.imshow('yuv', yuv)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


K = 8


trackbar_changed = False


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


def quantization_kmeans(Z, K, criteria, image):
    global color_palette
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # for i in range(K):
    #     cluster_points = Z[label.flatten() == i]
    #     plt.scatter(cluster_points[:, 0],
    #                 cluster_points[:, 1], label=f'Cluster {i + 1}')

    # # Plot the cluster centers
    # plt.scatter(center[:, 0], center[:, 1], marker='X',
    #             s=200, c='red', label='Cluster Centers')

    # plt.title('K-Means Clustering Results')
    # plt.legend()
    # plt.show()

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    color_palette = center.copy()
    # convert color_palette to int
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
    global video_window
    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 80)

    cv2.createTrackbar('K', 'Trackbars', K, 64, on_trackbar_change)

    Z = image.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    video_window.cargar_frame(image)
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

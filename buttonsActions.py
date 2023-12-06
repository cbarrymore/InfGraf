import os
from PySide2.QtCore import QCoreApplication, QRect, QMetaObject, Qt, QUrl
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QMenuBar, QToolBar, QStatusBar, QFileDialog, QLabel
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
import cv2
import pytorch_media_detect
import torch
import numpy as np
import counting_by_color


class ArchivoInfo:
    def __init__(self):
        self.nombre = ""
        self.tipo = ""


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

archivo = ArchivoInfo()
count_frame = 0
color_count = {}
color_search = np.zeros((200, 200, 3), np.uint8)
color_selected = np.zeros((200, 200, 3), np.uint8)
hue = 0
mouse_callback_triggered = False


def action_cargar_imagen(self):
    filename, _ = QFileDialog.getOpenFileName(
        self, "Seleccionar imagen", ".", "Images (*.png *.jpg *.bmp)")
    if filename:
        pixmap = QPixmap(filename)
        crear_nuevo_archivo(str(filename))
        self.label_2.setText(filename)
        self.label_3.setPixmap(pixmap)


def action_cargar_video(self):
    filtro_video = "Archivos de video (*.avi *.mp4 *.mkv *.mov);;Todos los formatos de video (*.avi *.mp4 *.mkv *.mov);;Otros (*.*)"
    nombre, _ = QFileDialog.getOpenFileName(
        self, "Abrir video", ".", filtro_video)
    if nombre:
        crear_nuevo_archivo(str(nombre))
        self.label_2.setText(nombre)


def action_mostrar_contenido(self):
    global count_frame

    if archivo.tipo == "video":
        cap = cv2.VideoCapture(archivo.nombre)

        ret, frame = cap.read()

        # Crea una ventana con un nombre
        cv2.namedWindow("Video")

        count_frame = 0

        while cap.isOpened():
            if self.pause == True:
                cv2.imshow("Video", frame)
                if self.counting == True:
                    frame_str = "frame%d.jpg" % count_frame
                    cv2.imwrite(frame_str, frame)
                    do_image(frame_str)
                    os.remove(frame_str)
                    self.counting = False

            else:
                ret, frame = cap.read()
                if not ret:
                    break
                print("tratando frame")
                if self.object_detection == True:
                    # call a function to detect objects
                    frame = detect_objects(self, frame)
                cv2.imshow("Video", np.squeeze(frame))
                count_frame += 1
            if cv2.waitKey(24) > 0:
                break
        cap.release()
    else:
        if self.counting == True:
            do_image(archivo.nombre)
        else:
            img = cv2.imread(archivo.nombre)
            if self.object_detection == True:
                # call a function to detect objects
                img = detect_objects(self, img)
            cv2.imshow("MainWindow", np.squeeze(img))


def crear_nuevo_archivo(nombre):
    archivo.nombre = nombre
    if archivo.nombre.split('.')[-1].lower() in ["jpg", "png", "bmp"]:
        archivo.tipo = "imagen"
    else:
        archivo.tipo = "video"


def action_activate_color_clustering(self):
    if self.color_clustering == True:
        self.color_clustering = False
    else:
        self.color_clustering = True
    if archivo.tipo == "imagen":
        self.mostrar_contenido()


def action_activar_object_detection(self):
    if self.object_detection == True:
        self.object_detection = False
    else:
        self.object_detection = True
    if archivo.tipo == "imagen":
        self.mostrar_contenido()


def action_activate_counting(self):
    if self.counting == True:
        self.counting = False
    else:
        self.counting = True
    if archivo.tipo == "imagen":
        self.mostrar_contenido()


def detect_objects(self, frame):
    # Call the model to detect objects in the frame
    print("Carlos1")
    detection = self.model(frame)
    pred = detection.xyxy[0]  # img1 predictions (tensor)

    if self.color_clustering == True:
        print("Carlos2")
        detection_frame = func_color_clustering(pred, frame)
    elif self.object_detection:
        print("Carlos3")
        detection_frame = detection.render()[0]
    return detection_frame


def distancia_color(color1, color2):
    return sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)) ** 0.5


def get_center_of_detection(xmin, ymin, xmax, ymax, frame):
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
    return detection_center_image


def func_color_clustering(pred, frame):
    global color_count
    global count_frame
    k = 0
    pred = pred[pred[:, 5] == 0]
    margen = 100
    if count_frame % 5 == 0:
        color_count = {}
    for det in pred:
        k += 1
        xmin, ymin, xmax, ymax, conf, class_id = det
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        detection_center_image = get_center_of_detection(
            xmin, ymin, xmax, ymax, frame)
        cv2.imshow("detection_center_image", detection_center_image)
        cv2.imwrite(f"detection_center_image{k}.jpg", detection_center_image)

        cv2.waitKey(0)
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


def select_color(event, x, y, flags, param):
    global hue
    global color_search
    global color_selected
    global mouse_callback_triggered

    B = frame[y, x][0]
    G = frame[y, x][1]
    R = frame[y, x][2]
    color_search[:] = (B, G, R)

    if event == cv2.EVENT_LBUTTONDOWN:
        color_selected[:] = (B, G, R)
        hue = hsv[y, x][0]
        print(hsv[y, x])
    mouse_callback_triggered = True  # Se activa la variable global


def search_contours(mask, frame):
    contours_count = 0
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 10000:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            contours_count += 1

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            cv2.circle(frame, (cX, cY), 3, (255, 255, 255), -1)
            cv2.putText(frame, f"{contours_count}", (cX - 25, cY - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return contours_count


def nothing(x):
    pass


def do_image(filename):
    global frame
    global hsv
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_color)

    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 80)

    cv2.createTrackbar('Lower-Hue', 'Trackbars', 14, 179, nothing)
    cv2.createTrackbar('Upper-Hue', 'Trackbars', 18, 179, nothing)
    og_frame = cv2.imread(filename)
    while True:
        frame = og_frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        diff_lower_hue = cv2.getTrackbarPos('Lower-Hue', 'Trackbars')
        diff_upper_hue = cv2.getTrackbarPos('Upper-Hue', 'Trackbars')

        lower_hue = 0 if hue - diff_lower_hue < 0 else hue - diff_lower_hue
        upper_hue = hue + diff_upper_hue if hue + diff_upper_hue < 179 else 179

        lower_hsv = np.array([lower_hue, 50, 20])
        upper_hsv = np.array([upper_hue, 255, 255])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        count = search_contours(mask)

        cv2.putText(frame, f'Total: {count}', (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('mask', mask)
        cv2.imshow('image', frame)
        cv2.imshow('color_search', color_search)
        cv2.imshow('color_selected', color_selected)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def do_frame(frame_local):
    global hsv
    global frame

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_color)

    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 80)

    cv2.createTrackbar('Lower-Hue', 'Trackbars', 14, 179, nothing)
    cv2.createTrackbar('Upper-Hue', 'Trackbars', 18, 179, nothing)

    while True:
        frame = frame_local.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        diff_lower_hue = cv2.getTrackbarPos('Lower-Hue', 'Trackbars')
        diff_upper_hue = cv2.getTrackbarPos('Upper-Hue', 'Trackbars')

        lower_hue = 0 if hue - diff_lower_hue < 0 else hue - diff_lower_hue
        upper_hue = hue + diff_upper_hue if hue + diff_upper_hue < 179 else 179

        lower_hsv = np.array([lower_hue, 50, 20])
        upper_hsv = np.array([upper_hue, 255, 255])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        count = search_contours(mask)

        cv2.putText(frame, f'Total: {count}', (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('mask', mask)
        cv2.imshow('image', frame)
        cv2.imshow('color_search', color_search)
        cv2.imshow('color_selected', color_selected)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def action_activate_pause(self):
    if self.pause == True:
        self.pause = False
    else:
        self.pause = True


def do_video(frame_local):
    global frame
    frame = frame_local
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_color)

    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 80)

    cv2.createTrackbar('Lower-Hue', 'Trackbars', 14, 179, nothing)
    cv2.createTrackbar('Upper-Hue', 'Trackbars', 18, 179, nothing)
    while True:
        if mouse_callback_triggered:
            mouse_callback_triggered = False
            break
        cv2.imshow('image', frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        diff_lower_hue = cv2.getTrackbarPos('Lower-Hue', 'Trackbars')
        diff_upper_hue = cv2.getTrackbarPos('Upper-Hue', 'Trackbars')

        lower_hue = 0 if hue - diff_lower_hue < 0 else hue - diff_lower_hue
        upper_hue = hue + diff_upper_hue if hue + diff_upper_hue < 179 else 179

        lower_hsv = np.array([lower_hue, 50, 20])
        upper_hsv = np.array([upper_hue, 255, 255])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        count = search_contours(mask)

        cv2.putText(frame, f'Total: {count}', (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('mask', mask)
        cv2.imshow('image', frame)
        cv2.imshow('color_search', color_search)
        cv2.imshow('color_selected', color_selected)

    cv2.destroyAllWindows()
    return frame


def search_contours(mask):
    contours_count = 0
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 10000:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            contours_count += 1

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            cv2.circle(frame, (cX, cY), 3, (255, 255, 255), -1)
            cv2.putText(frame, f"{contours_count}", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        2)

    return contours_count

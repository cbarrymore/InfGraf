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
import quantization
import color


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
                color.do_image_HSV(frame)
                self.counting_HSV = False
            if self.counting_YUV == True:
                color.do_image_YUV(frame)
                self.counting_YUV = False
            if self.quantization == True:
                quantization.quantization(self, frame)
                # quantization(self, frame)
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
            color.do_image_HSV(img)
        if self.counting_YUV == True:
            color.do_image_YUV(img)
        if self.quantization == True:
            quantization.quantization(self, img)
            # quantization(self, img)
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

from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QFileDialog
import cv2
import numpy as np
from videoWindow import VideoWindow
import quantization
import color
from objectDetection import detect_objects


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
                frame = detect_objects(self, frame, count_frame)
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

def activate_quantization_roi(self):
    if not self.quantization_roi and self.object_detection:
        self.quantization_roi = True
    else:
        self.quantization_roi = False
    if archivo.tipo == "imagen":
        self.mostrar_contenido()

def action_pause(self):
    global cargada

    if self.pause == True:
        self.pause = False
    else:
        self.pause = True
        cargada = False
    if archivo.tipo == "imagen":
        self.mostrar_contenido()
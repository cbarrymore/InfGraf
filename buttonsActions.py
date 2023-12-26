from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QFileDialog
import cv2
from videoWindow import VideoWindow
import quantization
import color
from objectDetection import detect_objects


class ArchivoInfo:
    def __init__(self):
        self.nombre = ""
        self.tipo = ""

archivo = ArchivoInfo()
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
    global video_window

    if video_window == None or not video_window.isVisible():
        video_window = VideoWindow()
        video_window.show()

    if archivo.tipo == "video":
        cap = cv2.VideoCapture(archivo.nombre)

        ret, frame = cap.read()

        count_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.object_detection == True:
                frame = detect_objects(self, frame,archivo.tipo, video_window, count_frame)
            if self.counting_HSV == True:
                color.do_image_HSV(frame)
                self.counting_HSV = False
            if self.counting_YUV == True:
                color.do_image_YUV(frame)
                self.counting_YUV = False
            if self.quantization == True:
                quantization.quantization(self, frame)
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
        if not self.counting_HSV and not self.counting_YUV and not self.quantization:
            if self.object_detection == True:
                img = detect_objects(self, img, archivo.tipo, video_window)
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
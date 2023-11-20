################################################################################
# Form generated from reading UI file 'mainwindow.ui'
##
# Created by: Qt User Interface Compiler version 5.15.2
##
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import QCoreApplication, QRect, QMetaObject, Qt, QUrl
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QMenuBar, QToolBar, QStatusBar, QFileDialog, QLabel
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
import cv2
import pytorch_media_detect
import torch
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(431, 331)
        self.centralWidget = QWidget(MainWindow)
        self.centralWidget.setObjectName(u"centralWidget")
        self.pushButton = QPushButton(self.centralWidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(-250, 190, 75, 23))
        self.pushButton_2 = QPushButton(self.centralWidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(10, 10, 101, 31))
        self.pushButton_3 = QPushButton(self.centralWidget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(120, 10, 101, 31))
        self.label = QLabel(self.centralWidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 220, 47, 13))
        self.label_2 = QLabel(self.centralWidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(70, 220, 351, 20))
        self.pushButton_4 = QPushButton(self.centralWidget)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(10, 240, 101, 31))
        self.pushButton_5 = QPushButton(self.centralWidget)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(120, 240, 101, 31))
        self.pushButton_6 = QPushButton(self.centralWidget)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setGeometry(QRect(150, 280, 141, 71))
        self.pushButton_7 = QPushButton(self.centralWidget)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(150, 350, 141, 71))
        self.label_3 = QLabel(self.centralWidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(70, 50, 281, 151))
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 431, 21))
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QToolBar(MainWindow)
        self.mainToolBar.setObjectName(u"mainToolBar")
        MainWindow.addToolBar(Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QStatusBar(MainWindow)
        self.statusBar.setObjectName(u"statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate(
            "MainWindow", u"MainWindow", None))
        self.pushButton.setText(QCoreApplication.translate(
            "MainWindow", u"PushButton", None))
        self.pushButton_2.setText(QCoreApplication.translate(
            "MainWindow", u"Cargar imagen", None))
        self.pushButton_3.setText(QCoreApplication.translate(
            "MainWindow", u"Cargar video", None))
        self.label.setText(QCoreApplication.translate(
            "MainWindow", u"Nombre:", None))
        self.label_2.setText("")
        self.pushButton_4.setText(QCoreApplication.translate(
            "MainWindow", u"Ver contenido", None))
        self.pushButton_5.setText(
            QCoreApplication.translate("MainWindow", u"Detectar", None))
        self.label_3.setText("")
        self.pushButton_6.setText(
            QCoreApplication.translate("MainWindow", u"Colores", None))
        self.pushButton_7.setText(
            QCoreApplication.translate("MainWindow", u"Contar", None))
    # retranslateUi


class ArchivoInfo:
    def __init__(self):
        self.nombre = ""
        self.tipo = ""


archivo = ArchivoInfo()


class MyMainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.cargar_imagen)
        self.pushButton_3.clicked.connect(self.cargar_video)
        self.pushButton_4.clicked.connect(self.mostrar_contenido)
        self.pushButton_5.clicked.connect(
            self.activar_object_detection)
        self.pushButton_6.clicked.connect(self.activate_color_clustering)
        self.pushButton_7.clicked.connect(self.activate_counting)
        self.color_clustering = False
        self.object_detection = False
        self.counting = False
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.cuda()

    def cargar_imagen(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", ".", "Images (*.png *.jpg *.bmp)")
        if filename:
            pixmap = QPixmap(filename)
            self.crear_nuevo_archivo(str(filename))
            self.label_2.setText(filename)
            self.label_3.setPixmap(pixmap)

    def cargar_video(self):
        filtro_video = "Archivos de video (*.avi *.mp4 *.mkv *.mov);;Todos los formatos de video (*.avi *.mp4 *.mkv *.mov);;Otros (*.*)"
        nombre, _ = QFileDialog.getOpenFileName(
            self, "Abrir video", ".", filtro_video)
        if nombre:
            self.crear_nuevo_archivo(str(nombre))
            self.label_2.setText(nombre)

    def mostrar_contenido(self):
        if archivo.tipo == "video":
            cap = cv2.VideoCapture(archivo.nombre)
            while cap.isOpened():
                ret, frame = cap.read()
                print("tratando frame")
                if not ret:
                    break
                if self.object_detection == True:
                    # call a function to detect objects
                    frame = self.detect_objects(frame)
                if self.counting == True:
                    frame = self.counting(frame)
                # if color_principal == True:
                cv2.imshow("MainWindow", np.squeeze(frame))
                if cv2.waitKey(1) > 0:
                    break
            cap.release()
        else:
            img = cv2.imread(archivo.nombre)
            if self.object_detection == True:
                # call a function to detect objects
                img = self.detect_objects(img)
            cv2.imshow("MainWindow", np.squeeze(img))

    def crear_nuevo_archivo(self, nombre):
        archivo.nombre = nombre
        if archivo.nombre.split('.')[-1].lower() in ["jpg", "png", "bmp"]:
            archivo.tipo = "imagen"
        else:
            archivo.tipo = "video"

    def activate_color_clustering(self):
        if self.color_clustering == True:
            self.color_clustering = False
        else:
            self.color_clustering = True
        if archivo.tipo == "imagen":
            self.mostrar_contenido()

    def activar_object_detection(self):
        if self.object_detection == True:
            self.object_detection = False
        else:
            self.object_detection = True
        if archivo.tipo == "imagen":
            self.mostrar_contenido()

    def activate_counting(self):
        if self.counting == True:
            self.counting = False
        else:
            self.counting = True
        if archivo.tipo == "imagen":
            self.mostrar_contenido()

    def counting(self, frame):
        
    
    def detect_objects(self, frame):
        # Call the model to detect objects in the frame
        print("Carlos1")
        detection = self.model(frame)
        pred = detection.xyxy[0]  # img1 predictions (tensor)

        if self.object_detection:
            print("Carlos3")
            detection_frame = detection.render()[0]
        if self.color_clustering == True:
            print("Carlos2")
            detection_frame = self.func_color_clustering(pred, frame)
        return detection_frame

    def func_color_clustering(self, pred, frame):
        pred = pred[pred[:, 5] == 0]
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
            number_clusters = 1
            compactness, labels, centers = cv2.kmeans(
                data, number_clusters, None, criteria, 10, flags)
            print(centers)
            for index, row in enumerate(centers):
                blue, green, red = int(row[0]), int(row[1]), int(row[2])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                              (blue, green, red), 2)
        return frame

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

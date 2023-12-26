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
import counting_by_color
import numpy as np
import buttonsActions


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(406, 390)
        self.centralWidget = QWidget(MainWindow)
        self.centralWidget.setObjectName(u"centralWidget")
        self.pushButton = QPushButton(self.centralWidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(-250, 190, 75, 23))
        self.pushButton_2 = QPushButton(self.centralWidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(20, 10, 161, 31))
        self.pushButton_3 = QPushButton(self.centralWidget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(200, 10, 161, 31))
        self.pushButton_4 = QPushButton(self.centralWidget)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(40, 170, 101, 31))
        self.pushButton_5 = QPushButton(self.centralWidget)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(40, 230, 101, 31))
        self.pushButton_6 = QPushButton(self.centralWidget)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setGeometry(QRect(140, 50, 101, 31))
        self.pushButton_7 = QPushButton(self.centralWidget)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(140, 90, 101, 31))
        self.label = QLabel(self.centralWidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(50, 150, 91, 16))
        self.label_3 = QLabel(self.centralWidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(30, 210, 131, 16))
        self.label_4 = QLabel(self.centralWidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(220, 150, 151, 20))
        self.pushButton_8 = QPushButton(self.centralWidget)
        self.pushButton_8.setObjectName(u"pushButton_8")
        self.pushButton_8.setGeometry(QRect(240, 170, 101, 31))
        self.pushButton_9 = QPushButton(self.centralWidget)
        self.pushButton_9.setObjectName(u"pushButton_9")
        self.pushButton_9.setGeometry(QRect(240, 230, 101, 31))
        self.label_2 = QLabel(self.centralWidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(130, 280, 141, 20))
        self.pushButton_10 = QPushButton(self.centralWidget)
        self.pushButton_10.setObjectName(u"pushButton_10")
        self.pushButton_10.setGeometry(QRect(90, 300, 101, 31))
        self.pushButton_11 = QPushButton(self.centralWidget)
        self.pushButton_11.setObjectName(u"pushButton_11")
        self.pushButton_11.setGeometry(QRect(200, 300, 101, 31))
        self.label_3 = QLabel(self.centralWidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(120, 340, 151, 16))
        self.pushButton_12 = QPushButton(self.centralWidget)
        self.pushButton_12.setObjectName(u"pushButton_12")
        self.pushButton_12.setGeometry(QRect(140, 360, 101, 31))

        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 406, 21))
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
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Cargar imagen", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Cargar video", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"Detectar", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"Color", None))
        self.pushButton_6.setText(QCoreApplication.translate("MainWindow", u"Ver contenido", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"Pausar", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Detectar objetos:", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Conteo de objetos (RGB)", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Detecci\u00f3n de objetos por color", None))
        self.pushButton_8.setText(QCoreApplication.translate("MainWindow", u"Formato HSV", None))
        self.pushButton_9.setText(QCoreApplication.translate("MainWindow", u"Formato YUV", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Otras implementaciones", None))
        self.pushButton_10.setText(QCoreApplication.translate("MainWindow", u"Cuantificaci\u00f3n", None))
        self.pushButton_11.setText(QCoreApplication.translate("MainWindow", u"Dithering", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Cuantificación de detecciones", None))
        self.pushButton_12.setText(QCoreApplication.translate("MainWindow", u"Cuantificaci\u00f3n", None))
    # retranslateUi


class MyMainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.cargar_imagen)
        self.pushButton_3.clicked.connect(self.cargar_video)
        self.pushButton_4.clicked.connect(self.activar_object_detection)
        self.pushButton_5.clicked.connect(self.activate_color_counting)
        self.pushButton_6.clicked.connect(self.mostrar_contenido)
        self.pushButton_7.clicked.connect(self.activate_pause)
        self.pushButton_8.clicked.connect(self.activate_counting_HSV)
        self.pushButton_9.clicked.connect(self.activate_counting_YUV)
        self.pushButton_10.clicked.connect(self.activate_quantization)
        self.pushButton_11.clicked.connect(self.activate_dithering)
        self.pushButton_12.clicked.connect(self.activate_quantization_roi)

        self.color_clustering = False
        self.object_detection = False
        self.color_counting = False
        self.counting_HSV = False
        self.counting_YUV = False
        self.quantization = False
        self.dithering = False
        self.quantization_roi = False
        self.pause = False
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.cuda()

    def cargar_imagen(self):
        buttonsActions.action_cargar_imagen(self)

    def cargar_video(self):
        buttonsActions.action_cargar_video(self)

    def mostrar_contenido(self):
        buttonsActions.action_mostrar_contenido(self)

    def activate_color_counting(self):
        buttonsActions.action_activate_color_counting(self)

    def activar_object_detection(self):
        buttonsActions.action_activar_object_detection(self)

    def activate_counting_HSV(self):
        buttonsActions.action_activate_counting_HSV(self)

    def activate_counting_YUV(self):
        buttonsActions.action_activate_counting_YUV(self)

    def activate_quantization(self):
        buttonsActions.action_activate_quantization(self)

    def activate_dithering(self):
        buttonsActions.action_activate_dithering(self)

    def activate_quantization_roi(self):
        buttonsActions.activate_quantization_roi(self)

    def activate_pause(self):
        buttonsActions.action_pause(self)

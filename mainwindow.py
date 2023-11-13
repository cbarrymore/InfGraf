# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import QCoreApplication, QRect, QMetaObject, Qt
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QMenuBar, QToolBar, QStatusBar, QFileDialog, QLabel
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import cv2
import pytorch_media_detect


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
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Cargar imagen", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Cargar video", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Nombre:", None))
        self.label_2.setText("")
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"Detectar", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"Color", None))
        self.label_3.setText("")
    # retranslateUi

class ArchivoInfo:
    def __init__(self):
        self.nombre = ""
        
archivo = ArchivoInfo()

class MyMainWindow(QMainWindow, Ui_MainWindow):    
    
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.cargar_imagen)
        self.pushButton_3.clicked.connect(self.cargar_video)
        self.pushButton_4.clicked.connect(self.detectar_objetos)

    def cargar_imagen(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", ".", "Images (*.png *.jpg *.bmp)")
        if filename:
            pixmap = QPixmap(filename)
            self.crear_nuevo_archivo(str(filename))
            self.label_2.setText(filename)
            self.label_3.setPixmap(pixmap)

    def cargar_video(self):
        filtro_video = "Archivos de video (*.avi *.mp4 *.mkv *.mov);;Todos los formatos de video (*.avi *.mp4 *.mkv *.mov);;Otros (*.*)"
        nombre, _ = QFileDialog.getOpenFileName(self, "Abrir video", ".", filtro_video)
        if nombre:
            self.crear_nuevo_archivo(str(nombre))         
            media_player = QMediaPlayer()
            media_content = QMediaContent(QUrl.fromLocalFile(nombre))
            media_player.setMedia(media_content)

            pixmap = QPixmap(media_player.thumbnail())
            self.label_2.setText(nombre)
            self.label_3.setPixmap(pixmap.scaledToWidth(300))
            
    def detectar_objetos(self):
        pytorch_media_detect.detectar_objetos(archivo.nombre)
            
    def crear_nuevo_archivo(self, nombre):
        archivo.nombre = nombre

    #def detectar_color_principal


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
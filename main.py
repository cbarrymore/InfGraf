from PySide2.QtWidgets import QApplication
from mainwindow import MyMainWindow


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
from PySide2.QtCore import Qt
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtWidgets import QLabel, QVBoxLayout, QWidget

class VideoWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self):
        super().__init__()
        self.resize(600, 500)
        self.setMinimumSize(200, 200)
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
    
    def cargar_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
      
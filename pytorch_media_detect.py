import torch
import cv2
import numpy as np
import sys
import os
import imghdr

path = "C:/Users/cbarr/Downloads/USMNT vs. Ghana _ Highlights_ October 17, 2023.mp4"
def detectar_objetos(ruta):
    output = "detection_video.mp4"
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.cuda()

    #if len(sys.argv) != 2:
    #    print("Uso: python programa.py <ruta o número>")
    #else:
    argumento = ruta
    if argumento.isnumeric():
        cap = cv2.VideoCapture(0)
        while 1:
            ret, frame = cap.read()
            if not ret:
                break
            detect = model(frame)
            cv2.imshow("img", np.squeeze(detect.render()))
            t = cv2.waitKey(50)
            if t == 27:
                break
    else:
        if os.path.exists(argumento):
            if imghdr.what(argumento) is not None:
                img = cv2.imread(argumento)
                detection = model(img)
                cv2.imshow("xd", np.squeeze(detection.render()))
                cv2.waitKey(0)
            elif argumento.endswith(".mp4"):
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                cap = cv2.VideoCapture(argumento)
                width = int(cap.get(3))
                height = int(cap.get(4))
                out = cv2.VideoWriter(output, fourcc, 24.0, (width, height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    print("tratando frame")
                    if not ret:
                        break

                    detection = model(frame)
                    frame_with_detection = detection.render()[0]
                    out.write(frame_with_detection)
                    cv2.imshow("video", np.squeeze(frame_with_detection))
                    if cv2.waitKey(1) > 0:
                        break
        else:
            print("El argumento no es un número ni una ruta válida.")
# cap = cv2.VideoCapture(0)

# width = int(cap.get(3))
# height = int(cap.get(4))

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output, fourcc, 30.0, (width, height))

# while cap.isOpened():
#     ret,frame =  cap.read()
#     print("tratando frame")
#     if not ret:break

#     detection = model(frame)
#     frame_with_detection= detection.render()[0]
#     out.write(frame_with_detection)
#     cv2.imshow("video",np.squeeze(frame_with_detection))
#     if cv2.waitKey(1) > 0: break

# cap.release()  # Libera los recursos del video de entrada
# out.release()  # Cierra el video de salida
# cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV

# img = cv2.imread("D:/TFG/IAUSD-A 100k images dataset for automatic soccer analysis/IAUSD-A 100k images dataset for automatic soccer analysis/11.jpg")

# detection = model(img)

# cv2.imshow("xd",np.squeeze(detection.render()))
# cv2.waitKey(0)

# cap= cv2.VideoCapture(0)

# while 1:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     detect = model(frame)
#     cv2.imshow("img", np.squeeze(detect.render()))
#     t = cv2.waitKey(50)
#     if t == 27: break

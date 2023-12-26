import numpy as np
import cv2

color_count = {}
K = 3
create_Trackbar = False

def detect_objects(self, frame, count_frame = 0):
    original_frame = frame.copy()
    detection = self.model(frame)
    pred = detection.xyxy[0]  # frame predictions (tensor)

    if self.object_detection:
        detection_frame = detection.render()[0]
    if self.color_counting:
        detection_frame = func_color_counting(pred, frame, original_frame, count_frame)
    if self.quantization_roi:
        detection_frame = quantization_detections(pred, frame, original_frame)
    return detection_frame

def distancia_color(color1, color2):
    return sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)) ** 0.5

def func_color_counting(pred, frame, original_frame, count_frame = 0):
    global color_count

    pred = pred[pred[:, 5] == 0]
    margen = 100
    if count_frame % 5 == 0:
        color_count = {}
    for det in pred:
        xmin, ymin, xmax, ymax, conf, class_id = det
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        # Obtenemos el centro del rectangulo de detección
        x_center_of_rectangle, y_center_of_rectangle = int((xmax+xmin)//2), int((ymax+ymin)//2)
        # Sacamos la anchura y altura que queremos que tenga la imagen del centro del rectangulo de deteccion
        center_width_min, center_width_max = (x_center_of_rectangle + xmin)//2, (x_center_of_rectangle+xmax)//2
        center_height_min, center_height_max = (y_center_of_rectangle + ymin)//2, (y_center_of_rectangle)
        # Sacamos de la imagen original la imagen del centro del rectangulo de deteccion
        detection_center_image = original_frame[center_height_min:center_height_max, center_width_min:center_width_max]
        # Sacamos las dimensiones del centro de deteccion
        height, width, _ = np.shape(detection_center_image)

        data = np.reshape(detection_center_image, (height * width, 3))
        data = np.float32(data)
        median = np.mean(data, axis=0)        
        blue, green, red = int(median[0]), int(median[1]), int(median[2])
        color_encontrado = None
        distancia_minima = float('inf')
        
        if count_frame % 5 == 0:
            for color_key in color_count:
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
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (blue, green, red), 2)
    
    i = 0
    for color, count in color_count.items():
        cv2.putText(frame, f"Numero total: {count}", (0 + i, 0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color),
                    1)
        i += 150

    return frame

def on_trackbar_change(value):
    global K
    K = value

def quantization_detections(pred, frame, original_frame):
    global create_Trackbar
    global K
    
    if not create_Trackbar:
        cv2.namedWindow('Trackbars')
        cv2.resizeWindow('Trackbars', 400, 80)
        cv2.createTrackbar('K', 'Trackbars', K, 64, on_trackbar_change)
        create_Trackbar= True
        
    pred = pred[pred[:, 5] == 0]      
    for det in pred:      
        xmin, ymin, xmax, ymax, conf, class_id = det
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])  
        roi = original_frame[ymin:ymax, xmin:xmax]
        roi = quantization_roi(roi)          
        frame[ymin:ymax, xmin:xmax] = roi
        
    return frame
            
def quantization_roi(roi):
    global K
    
    Z = roi.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    if K == 0:
        K = 1
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    quantization_roi = center[label.flatten()]   
    quantization_roi = quantization_roi.reshape(roi.shape)
    
    return quantization_roi
                    
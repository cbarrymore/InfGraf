import cv2
import numpy as np
from videoWindow import VideoWindow

color_selected = np.zeros((200, 200, 3), np.uint8)
color_search = np.zeros((200, 200, 3), np.uint8)
hue = 0
saturation = 0
value = 0
mouse_callback_triggered = False
white_flag = False
black_flag = False
font = cv2.FONT_HERSHEY_SIMPLEX


def search_contours(mask):
    contours_count = 0
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            contours_count += 1
    return contours_count


################### Formato HSV ################################

def select_color_HSV(event, x, y, flags, param):
    global hue
    global saturation
    global value
    global white_flag
    global black_flag

    B = frame[y, x][0]
    G = frame[y, x][1]
    R = frame[y, x][2]
    color_search[:] = (B, G, R)

    if event == cv2.EVENT_LBUTTONDOWN:
        white_flag = False
        black_flag = False
        color_selected[:] = (B, G, R)
        hue = hsv[y, x][0]
        saturation = hsv[y, x][1]
        value = hsv[y, x][2]
        if saturation <= 40 and value >= 100:
            white_flag = True
        elif value <= 40:
            black_flag = True

        print(hsv[y, x])
    mouse_callback_triggered = True  # Se activa la variable global


def do_image_HSV(image):
    global frame
    global hsv
    global video_window

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_color_HSV)

    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 80)

    cv2.createTrackbar('Lower-Hue', 'Trackbars', 14, 179, lambda x: None)
    cv2.createTrackbar('Upper-Hue', 'Trackbars', 18, 179, lambda x: None)
    video_window = VideoWindow()
    video_window.show()
    while True:
        frame = image.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        diff_lower_hue = cv2.getTrackbarPos('Lower-Hue', 'Trackbars')
        diff_upper_hue = cv2.getTrackbarPos('Upper-Hue', 'Trackbars')

        lower_hue = 0 if hue - diff_lower_hue < 0 else hue - diff_lower_hue
        upper_hue = hue + diff_upper_hue if hue + diff_upper_hue < 179 else 179

        lower_saturation = 50
        upper_saturation = 255
        lower_value = 20
        upper_value = 255

        if white_flag:
            lower_saturation = 0
            upper_saturation = 40
            lower_value = 100
            upper_value = 255
        elif black_flag:
            lower_saturation = 0
            upper_saturation = 255
            lower_value = 0
            upper_value = 40
        if white_flag or black_flag:
            lower_hue = 0
            upper_hue = 179
        lower_hsv = np.array([lower_hue, lower_saturation, lower_value])
        upper_hsv = np.array([upper_hue, upper_saturation, upper_value])

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        count = search_contours(mask)

        cv2.putText(frame, f'Total: {count}', (5, 30),
                    font, 1, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('mask', mask)
        cv2.imshow('image', frame)
        video_window.cargar_frame(frame)
        # cv2.imshow('image', frame)
        cv2.imshow('color_search', color_search)
        cv2.imshow('color_selected', color_selected)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


################### Formato YUV ################################
_y = 0
u = 0
v = 0
u_v_positive = False
u_v_negative = False
u_positive_v_negative = False
u_negative_v_positive = False
near_center = False
white = False


def select_color_YUV(event, x, y, flags, param):
    global hue
    global _y
    global u
    global mouse_callback_triggered
    global u_v_positive
    global u_v_negative
    global u_positive_v_negative
    global u_negative_v_positive
    global near_center
    global white
    global v
    B = frame[y, x][0]
    G = frame[y, x][1]
    R = frame[y, x][2]
    color_search[:] = (B, G, R)

    if event == cv2.EVENT_LBUTTONDOWN:
        color_selected[:] = (B, G, R)
        _y = yuv[y, x][0]
        u = yuv[y, x][1]
        v = yuv[y, x][2]
        # check if u and v are close to 127 by a margin of 10
        if (_y >= 127 or _y < 127) and abs(u - 127) <= 10 and abs(v - 127) <= 10:
            near_center = True
            # if _y -64 < 127 by a margin of 10
            white = _y > 127

        if u > 127 and v > 127:
            u_v_positive = True
        elif u < 127 and v > 127:
            u_negative_v_positive = True
        elif u > 127 and v < 127:
            u_positive_v_negative = True
        else:
            u_v_negative = True
        print(yuv[y, x])
        mouse_callback_triggered = True  # Se activa la variable global


def filter_colors_YUV(yuv_image, radius):
    global u
    global v
    distances = np.sqrt(((yuv_image[:, :, 0] - _y) ** 2 + yuv_image[:, :, 1] - u) ** 2 +
                        (yuv_image[:, :, 2] - v) ** 2)

    # Crear una máscara para los píxeles dentro del radio especificado
    mask = distances <= radius

    return mask


def do_image_YUV(image):
    global frame
    global yuv
    global mouse_callback_triggered
    global near_center
    global white
    global u_v_positive
    global u_v_negative
    global u_positive_v_negative
    global u_negative_v_positive
    global video_window

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_color_YUV)

    frame = image.copy()
    mask = None
    video_window = VideoWindow()
    video_window.show()
    while True:
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        if mouse_callback_triggered:
            frame = image.copy()
            if near_center:
                if white:
                    print("white")
                    mask = np.logical_and(
                        yuv[:, :, 0] > 127, np.logical_and(abs(yuv[:, :, 1] - 127) <= 10, abs(yuv[:, :, 2] - 127) <= 10))
                    # if abs(u - 127) <= 10 and abs(v - 127) <= 10:
                else:
                    print("black")
                    mask = np.logical_and(
                        yuv[:, :, 0] < 127, np.logical_and(abs(yuv[:, :, 1] - 127) <= 20, abs(yuv[:, :, 2] - 127) <= 20))
            else:
                if u_v_positive:
                    mask = np.logical_and(
                        yuv[:, :, 1] > 127, yuv[:, :, 2] > 127)
                elif u_v_negative:
                    mask = np.logical_and(
                        yuv[:, :, 1] < 127, yuv[:, :, 2] < 127)
                elif u_positive_v_negative:
                    mask = np.logical_and(
                        yuv[:, :, 1] > 127, yuv[:, :, 2] < 127)
                else:
                    mask = np.logical_and(
                        yuv[:, :, 1] < 127, yuv[:, :, 2] > 127)
            mouse_callback_triggered = False
            white = False
            near_center = False
            u_v_negative = False
            u_v_positive = False
            u_positive_v_negative = False
            u_negative_v_positive = False
            mask = mask.astype(np.uint8) * 255
            count = search_contours(mask)
            cv2.putText(frame, f'Total: {count}', (5, 30),
                        font, 1, (255, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('mask', mask)
        video_window.cargar_frame(frame)
        cv2.imshow('image', frame)
        cv2.imshow('color_search', color_search)
        cv2.imshow('color_selected', color_selected)
        cv2.imshow('yuv', yuv)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

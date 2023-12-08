import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

color_search = np.zeros((200, 200, 3), np.uint8)
color_selected = np.zeros((200, 200, 3), np.uint8)

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

mouse_callback_triggered = False  # Nueva variable global


def select_color(event, x, y, flags, param):
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
        if abs(u - 127) <= 30 and abs(v - 127) <= 30:
            near_center = True
            print("near_center: true")
            # if _y -64 < 127 by a margin of 10
            white = _y > 80
            print("white: ", white)
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


def search_contours(mask):
    contours_count = 0
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 10000:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            contours_count += 1

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            cv2.circle(frame, (cX, cY), 3, (255, 255, 255), -1)
            cv2.putText(frame, f"{contours_count}", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        2)

    return contours_count


def nothing(x):
    pass


def filter_colors_yuv(yuv_image, radius):
    global u
    global v
    distances = np.sqrt(((yuv_image[:, :, 0] - _y) ** 2 + yuv_image[:, :, 1] - u) ** 2 +
                        (yuv_image[:, :, 2] - v) ** 2)

    # Crear una máscara para los píxeles dentro del radio especificado
    mask = distances <= radius

    return mask


def do_image(filename):
    global frame
    global yuv
    global mouse_callback_triggered
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', select_color)

    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 80)

    cv2.createTrackbar('Lower-u', 'Trackbars', 14, 179, nothing)
    cv2.createTrackbar('Upper-u', 'Trackbars', 18, 179, nothing)

    cv2.createTrackbar('Lower-v', 'Trackbars', 14, 179, nothing)
    cv2.createTrackbar('Upper-v', 'Trackbars', 18, 179, nothing)

    cv2.namedWindow('Trackbars_v')
    cv2.resizeWindow('Trackbars_v', 400, 80)

    cv2.createTrackbar('Lower-v', 'Trackbars_v', 14, 179, nothing)
    cv2.createTrackbar('Upper-v', 'Trackbars_v', 18, 179, nothing)

    og_frame = cv2.imread(filename)
    while True:
        frame = og_frame.copy()
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        if mouse_callback_triggered:
            if near_center:
                if white:
                    mask = np.logical_and(
                        abs(yuv[:, :, 1] - 127) <= 30, abs(yuv[:, :, 2] - 127) <= 30)
                    # if abs(u - 127) <= 10 and abs(v - 127) <= 10:
                elif not white:
                    mask = np.logical_and(
                        abs(yuv[:, :, 1] - 127) <= 30, abs(yuv[:, :, 2] - 127) <= 30)
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
            mask = mask.astype(np.uint8) * 255
            print(mask)
            
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            count = search_contours(mask)

            cv2.putText(frame, f'Total: {count}', (5, 30),
                        font, 1, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('mask', mask)
            cv2.imshow('image', frame)
            cv2.waitKey(0)
            quit()

        diff_lower_u = cv2.getTrackbarPos('Lower-u', 'Trackbars')
        diff_upper_u = cv2.getTrackbarPos('Upper-u', 'Trackbars')

        diff_lower_v = cv2.getTrackbarPos('Lower-v', 'Trackbars_v')
        diff_upper_v = cv2.getTrackbarPos('Upper-v', 'Trackbars_v')

        lower_u = 0 if u - diff_lower_u < 0 else u - diff_lower_u
        upper_u = u + diff_upper_u if u + diff_upper_u < 255 else 255

        lower_v = 0 if v - diff_lower_v < 0 else u - diff_lower_v
        upper_v = v + diff_upper_v if v + diff_upper_v < 255 else 255

        # print (lower_u, upper_u, lower_v, upper_v) with ("lower_u = ...")
        if mouse_callback_triggered:
            print("u= ", u, "v = ", v)
            print("lower_u = ", lower_u, "upper_u = ", upper_u)
            print("lower_v = ", lower_v, "upper_v = ", upper_v)
            mouse_callback_triggered = False

        lower_uv_yuv = np.array([0, lower_u, lower_v])
        upper_uv_yuv = np.array([255, upper_u, upper_v])

        u_mask = cv2.inRange(yuv, lower_uv_yuv, upper_uv_yuv)

        v_mask = cv2.inRange(yuv, lower_uv_yuv, upper_uv_yuv)

        # cv2.bitwise_and(u_mask, v_mask, mask)
        # count = search_contours(mask)

        # cv2.putText(frame, f'Total: {count}', (5, 30),
        #             font, 1, (255, 0, 255), 2, cv2.LINE_AA)

        # cv2.imshow('mask_u', u_mask)
        # cv2.imshow('mask_v', v_mask)
        cv2.imshow('image', frame)
        cv2.imshow('color_search', color_search)
        cv2.imshow('color_selected', color_selected)
        cv2.imshow('yuv', yuv)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


cv2.destroyAllWindows()

if __name__ == '__main__':
    do_image("Soccer-EURO-2020-770x513.jpg")
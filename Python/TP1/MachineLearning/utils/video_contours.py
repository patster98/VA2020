import cv2
import numpy as np

window_name = 'contourImg'

trackbar_name = 'ThreshVal'
trackbar3_name = 'MaxContArea'
trackbar4_name = 'MinContArea'

def on_trackbar_change(val):
    # do nothing
    pass

def createTrackbars():
    cv2.namedWindow(window_name)

    slider_max = 200
    slider2_max = 100

    cv2.createTrackbar(trackbar_name, window_name, 150, slider_max, on_trackbar_change)
    cv2.createTrackbar(trackbar3_name, window_name, 8000, 100000, on_trackbar_change)
    cv2.createTrackbar(trackbar4_name, window_name, 8000, 10000, on_trackbar_change)


def binaryOPS(frame):

    threshVal_trackbar = cv2.getTrackbarPos(trackbar_name, window_name)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                int(threshVal_trackbar/2)*2+3, 2)
    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    # bin = 255 - bin  # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

    # kernel = np.ones((3, 3), np.uint8)  # Tamaño del bloque a recorrer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    # bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)
    opening1 = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
    closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
    return closing1

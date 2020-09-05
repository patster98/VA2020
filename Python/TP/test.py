import cv2
import numpy as np

def on_trackbar_change(val):
    # do nothing
    pass

window_name = 'Binary_manualThresh'
window_name2 = 'Binary_autoThresh'
trackbar_name = 'ThreshVal'
trackbar2_name = 'MaxThresh'
cv2.namedWindow(window_name)
cv2.namedWindow(window_name2)

slider_max = 255
slider2_max = 255
cv2.createTrackbar(trackbar_name, window_name, 1, slider_max, on_trackbar_change)
cv2.createTrackbar(trackbar2_name, window_name, 0, slider2_max, on_trackbar_change)
cv2.createTrackbar(trackbar2_name, window_name2, 0, slider2_max, on_trackbar_change)


def Tp():

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        _, frame2 = cap.read()


        threshVal_trackbar = cv2.getTrackbarPos(trackbar_name, window_name)
        maxThresh_trackbar = cv2.getTrackbarPos(trackbar2_name, window_name)
        maxThresh_trackbar2 = cv2.getTrackbarPos(trackbar2_name, window_name2)


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        _, thresh1 = cv2.threshold(gray, threshVal_trackbar, maxThresh_trackbar, cv2.THRESH_BINARY)
        _, threshAuto = cv2.threshold(gray2, 127, maxThresh_trackbar2, cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
        opening2 = cv2.morphologyEx(threshAuto, cv2.MORPH_OPEN, kernel)
        closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, kernel)

        contoursDenoise1, _ = cv2.findContours(closing1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contoursDenoise2, _ = cv2.findContours(closing2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cnt1 = contoursDenoise1[1]
        cnt2 = contoursDenoise2[1]

        # cnt = contours_denoise[0]
        # max_area = cv2.contourArea(cnt)
        #
        # # toma el contorno mas grande
        # for cont in contours_denoise:
        #     if cv2.contourArea(cont) > max_area:
        #         cnt = cont
        #         max_area = cv2.contourArea(cont)

        cv2.drawContours(frame, [cnt1], -1, (0, 0, 255), 6) #sobre threshMan
        cv2.drawContours(frame2, [cnt2], -1, (0, 255, 255), 6) #sobre threshAuto



        cv2.imshow(window_name, cv2.flip(frame, 1))
        cv2.imshow(window_name2, cv2.flip(frame2, 1))




        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

Tp()
import cv2
import numpy as np
from math import sqrt, copysign, log10

def on_trackbar_change(val):
    # do nothing
    pass

window_name = 'Binary_manualThresh'
window_name2 = 'Binary_automaticThresh'
trackbar_name = 'ThreshVal'
trackbar2_name = 'MaxThresh'
cv2.namedWindow(window_name)
cv2.namedWindow(window_name2)

slider_max = 200
slider2_max = 255
cv2.createTrackbar(trackbar_name, window_name, 50, slider_max, on_trackbar_change)
cv2.createTrackbar(trackbar2_name, window_name, 50, slider2_max, on_trackbar_change)
cv2.createTrackbar(trackbar2_name, window_name2, 50, slider2_max, on_trackbar_change)


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

        contDnis1 = cv2.findContours(closing1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contDnis2 = cv2.findContours(closing2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # sin este if se rompe el programa cuando el trackbar esta en 0
        if (threshVal_trackbar and maxThresh_trackbar and maxThresh_trackbar2) == 0:
            cv2.drawContours(frame, contDnis1, 0, (0, 0, 255), 6)  # sobre threshMan
            cv2.drawContours(frame2, contDnis2, 0, (0, 255, 255), 6)  # sobre threshAuto

        else:
            cntDnis1 = cv2.findContours(closing1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntDnis2 = cv2.findContours(closing2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # alternativa para el contorno mas grande
            cntDnis1 = cntDnis1[0] if len(cntDnis1) == 2 else cntDnis1[1]
            cntDnis1 = sorted(cntDnis1, key=cv2.contourArea, reverse=True)[0]
            cntDnis2 = cntDnis2[0] if len(cntDnis2) == 2 else cntDnis2[1]
            cntDnis2 = sorted(cntDnis2, key=cv2.contourArea, reverse=True)[0]

            moments1 = cv2.moments(cntDnis1)
            huMoments1 = cv2.HuMoments(moments1)
            moments2 = cv2.moments(cntDnis2)
            huMoments2 = cv2.HuMoments(moments2)
            for i in range(0, 7):
                if (huMoments1[i] and huMoments2[i]) != 0:
                    huMoments1[i] = -1 * copysign(1.0, huMoments1[i]) * log10(abs(huMoments2[i]))
                    huMoments2[i] = -1 * copysign(1.0, huMoments2[i]) * log10(abs(huMoments2[i]))
                print("Hu Moments Manual -> h" + str(i) + " " + str(huMoments1[i]))
                print("Hu Moments Automatic -> h" + str(i) + " " + str(huMoments2[i]))

            cv2.drawContours(frame, [cntDnis1], -1, (0, 0, 255), 6) #sobre threshMan
            cv2.drawContours(frame2, [cntDnis2], -1, (0, 255, 255), 6) #sobre threshAuto



        cv2.imshow(window_name, cv2.flip(frame, 1))
        cv2.imshow(window_name2, cv2.flip(frame2, 1))


        print(int(moments1["m00"]))
        cX1 = int(moments1["m10"] / moments1["m00"])
        cY1 = int(moments1["m01"] / moments1["m00"])
        print(cX1)
        print(cY1)
        print(int(moments2["m00"]))
        cX2 = int(moments2["m10"] / moments2["m00"])
        cY2 = int(moments2["m01"] / moments2["m00"])
        print(cX2)
        print(cY2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

Tp()
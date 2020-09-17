import cv2
import numpy as np


def on_trackbar_change(val):
    # do nothing
    pass


window_name = 'contourImg'
window_name2 = 'BinaryImg'

trackbar_name = 'ThreshVal'
trackbar2_name = 'MinError%'
trackbar3_name = 'MaxContArea'
trackbar4_name = 'MinContArea'

cv2.namedWindow(window_name)
cv2.namedWindow(window_name2)

slider_max = 200
slider2_max = 100

cv2.createTrackbar(trackbar_name, window_name, 150, slider_max, on_trackbar_change)
cv2.createTrackbar(trackbar2_name, window_name, 50, slider2_max, on_trackbar_change)
cv2.createTrackbar(trackbar3_name, window_name, 8000, 100000, on_trackbar_change)
cv2.createTrackbar(trackbar4_name, window_name, 800, 1500, on_trackbar_change)


def Tp():
    cap = cv2.VideoCapture(0)
    comp = None

    while True:
        filtered = []
        similar = []
        noSimilar = []
        _, frame = cap.read()

        threshVal_trackbar = cv2.getTrackbarPos(trackbar_name, window_name)
        minError = cv2.getTrackbarPos(trackbar2_name, window_name)
        maxCnt = cv2.getTrackbarPos(trackbar3_name, window_name)
        minCnt = cv2.getTrackbarPos(trackbar4_name, window_name)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, thresh1 = cv2.threshold(gray, threshVal_trackbar, 250, cv2.THRESH_BINARY)
        # _, thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshVal_trackbar, 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)

        contDnis1, _ = cv2.findContours(closing1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # sin este if se rompe el programa cuando el trackbar esta en 0
        if (threshVal_trackbar and minError) == 0:
            cv2.drawContours(frame, contDnis1, 0, (0, 0, 255), 6)

        else:
            for c in contDnis1:
                area = cv2.contourArea(c)
                if minCnt + 700 < area < maxCnt:
                    filtered.append(c)

            if filtered != []:
                # previene que rompa el programa cuando no encuentra ningún contorno que cumpla las specs
                cnt = filtered[0]
                max_area = cv2.contourArea(cnt)
                # toma el contorno mas grande de los filtrados
                for cont in filtered:
                    if cv2.contourArea(cont) > max_area:
                        cnt = cont
                        max_area = cv2.contourArea(cont)
                cv2.drawContours(frame, [cnt], -1, (255, 108, 181), 10, cv2.LINE_AA)

                moments = cv2.moments(cnt)
                cX1 = int(moments["m10"] / moments["m00"])
                cY1 = int(moments["m01"] / moments["m00"])
                huMoments = np.array(cv2.HuMoments(moments), dtype=float)
                logTransf = -1 * np.sign(huMoments) * np.log10(np.abs(huMoments))
                cv2.drawMarker(frame, (cX1, cY1), (0, 0, 0), cv2.MARKER_CROSS, markerSize=20, thickness=1, line_type=8)

            if comp != []:
                for filter in filtered:
                    err = cv2.matchShapes(filter, comp, cv2.CONTOURS_MATCH_I1, 0)
                    if err < minError / 100:
                        similar.append(filter)
                        mom1 = cv2.moments(filter)
                        cX1 = int(mom1["m10"] / mom1["m00"])
                        cY1 = int(mom1["m01"] / mom1["m00"])
                        cv2.putText(frame, "True", (cX1, cY1), cv2.FONT_ITALIC, 1, (0, 0, 0), 2, cv2.LINE_4)
                    else:
                        noSimilar.append(filter)
                        mom = cv2.moments(filter)
                        cX = int(mom["m10"] / mom["m00"])
                        cY = int(mom["m01"] / mom["m00"])
                        cv2.putText(frame, "False", (cX, cY), cv2.FONT_ITALIC, 1, (0, 0, 0), 2, cv2.LINE_4)

                cv2.drawContours(frame, noSimilar, -1, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.drawContours(frame, similar, -1, (0, 255, 0), 4, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        cv2.imshow(window_name2, cv2.flip(thresh1, 1))
        if cv2.waitKey(1) & 0xFF == ord('c'):
            comp = cnt
            print(comp)
            # guardo el valor de lo que quiero comparar con huMoments
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(err)
            # print(comp)
            # print("cnt Moments: ", Moments)
            # print("cnt HuMoments: ", huMoments)
            break


Tp()

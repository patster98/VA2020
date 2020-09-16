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
cv2.createTrackbar(trackbar4_name, window_name, 100, 1000, on_trackbar_change)


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

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        # kernel2 = cv2.getGaussianKernel(ksize=3)
        opening1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)

        contDnis1, _ = cv2.findContours(closing1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # sin este if se rompe el programa cuando el trackbar esta en 0
        if (threshVal_trackbar and minError) == 0:
            cv2.drawContours(frame, contDnis1, 0, (0, 0, 255), 6)

        else:
            for c in contDnis1:
                area = cv2.contourArea(c)
                if minCnt < area < maxCnt:
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
            huMoments = np.array(cv2.HuMoments(moments), dtype=float)
            logTransf = -1 * np.sign(huMoments) * np.log10(np.abs(huMoments))

            for filter in filtered:
                err = cv2.matchShapes(filter, cnt, cv2.CONTOURS_MATCH_I1, 0)
                if err < minError / 100 and cnt != []:
                    similar.append(filter)
                else:
                    noSimilar.append(filter)
            cv2.drawContours(frame, noSimilar, -1, (0, 0, 255), 4, cv2.LINE_AA)
            cv2.drawContours(frame, similar, -1, (0, 255, 0), 4, cv2.LINE_AA)

        cv2.imshow(window_name, cv2.flip(frame, 1))
        cv2.imshow(window_name2, cv2.flip(thresh1, 1))

        if cv2.waitKey(1) & 0xFF == ord('c'):
            # guardo el valor de lo que quiero comparar con huMoments
            comp = cnt
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # print(filtered)
            print(err)
            print(comp)
            break


Tp()

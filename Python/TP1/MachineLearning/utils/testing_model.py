import cv2
import numpy as np
import glob

from utils.hu_moments_generation import hu_moments_of_video, hu_moments_of_file
from utils.label_converters import int_to_label
from utils.video_contours import binaryOPS

window_name = 'contourImg'

trackbar_name = 'ThreshVal'
trackbar3_name = 'MaxContArea'
trackbar4_name = 'MinContArea'



def load_and_test(model, cap, img):
    if img == 1:
        files = glob.glob('../MachineLearning/shapes/testing/*')
        for f in files:
            hu_moments = hu_moments_of_file(f) # Genera los momentos de hu de los files de testing
            sample = np.array([hu_moments], dtype=np.float32) # numpy
            testResponse = model.predict(sample)[1] # Predice la clase de cada file

            #Lee la imagen y la imprime con un texto
            image = cv2.imread(f)
            image_with_text = cv2.putText(image, int_to_label(testResponse), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("result", image_with_text)
            cv2.waitKey(0)
    else:
        while True:
            filtered = []

            _, frame = cap.read()
            procVid = binaryOPS(frame)
            contours, hierarchy = cv2.findContours(procVid, cv2.RETR_LIST,
                                                      cv2.CHAIN_APPROX_SIMPLE)  # encuetra los contornos

            maxCnt = cv2.getTrackbarPos(trackbar3_name, window_name)
            minCnt = cv2.getTrackbarPos(trackbar4_name, window_name)

            if (cv2.getTrackbarPos(trackbar_name, window_name)) == 0:
                cv2.drawContours(frame, [contour], 0, (0, 0, 255), 10, cv2.LINE_AA)

            else:
                for c in contours:
                    area = cv2.contourArea(c)
                    if minCnt + 700 < area < maxCnt:
                        filtered.append(c)

                if filtered != []:
                    # previene que rompa el programa cuando no encuentra ningún contorno que cumpla las specs

                    for filter in filtered:
                        hu_moments, moments = hu_moments_of_video(filter)
                        cX1 = int(moments["m10"] / moments["m00"])
                        cY1 = int(moments["m01"] / moments["m00"])
                        sample = np.array([hu_moments], dtype=np.float32)  # numpy
                        testResponse = model.predict(sample)[1]  # Predice la clase de cada file
                        cv2.putText(frame, int_to_label(testResponse), (cX1, cY1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0),
                                    2, cv2.LINE_AA)
                    cv2.drawContours(frame, filtered, -1, (0, 255, 0), 4, cv2.LINE_AA)

                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # nothing
                break
        cap.release()
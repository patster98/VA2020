import cv2
import numpy as np
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([38, 86, 0])
        upper_blue = np.array([121, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        #mask = cv2.inRange(hsv, Scalar(0, 0, 0), Scalar(179,50, 100), 255, 0)
        cv2.imshow('holo', cv2.flip(hsv, 1))
        cv2.imshow('holo2', cv2.flip(mask, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

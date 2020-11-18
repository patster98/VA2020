import cv2
import numpy as np

def create_trackbar(trackbar_name, window_name, slider_max, value=0):
    cv2.createTrackbar(trackbar_name, window_name, value, slider_max, on_trackbarChange)


def on_trackbarChange(val):
    #do nothing
    pass


def get_trackbar_value(trackbar_name, window_name):
    #get trackbar value as odd number
    return int(cv2.getTrackbarPos(trackbar_name, window_name) / 2) * 2 + 3

def denoise(frame, noise_value):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (noise_value, noise_value))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def draw_contours(denoised, original, min_contour_area, max_contour_area):

    CNT = []
    contours, hierarchy = cv2.findContours(denoised, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if min_contour_area <= area <= max_contour_area:
            CNT.append(cnt)
            cv2.drawContours(original, [cnt], -1, (0, 165, 255), 3, cv2.LINE_AA)
    if len(CNT) > 0:
        CNT = CNT[0:1]
    return CNT



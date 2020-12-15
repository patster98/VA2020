from __future__ import print_function  # Python 2/3 compatibility
import cv2
import numpy as np
import imutils
from utils.imgProcess import denoise, create_trackbar, get_trackbar_value, draw_contours


# recorte de frame
# frametop = frame[:600]
# framebot = frame[600:]


def cntSelect(videoStream):
    ######################### Trackbars + Windows #####################
    window_name = 'CNTtoTrack'
    window_name2 = 'BinaryImg'
    cv2.namedWindow(window_name)
    cv2.namedWindow(window_name2)

    trackbar_Val = 'ThreshVal'
    trackbar2_Noise = 'NoiseVal'
    trackbar3_MaxC = 'MaxContArea'
    trackbar4_MinC = 'MinContArea'

    maxThresh = 255
    maxNoise = 20
    contourArea_max = 500
    contourArea_min = 500
    create_trackbar(trackbar_Val, window_name2, maxThresh, 180)
    create_trackbar(trackbar2_Noise, window_name2, maxNoise)
    create_trackbar(trackbar4_MinC, window_name2, contourArea_min, 450)
    create_trackbar(trackbar3_MaxC, window_name2, contourArea_max, 500)

    ####################### Main CNT Substraction ####################
    while True:
        threshVal = get_trackbar_value(trackbar_Val, window_name2)
        noiseVal = get_trackbar_value(trackbar2_Noise, window_name2)
        minCNT = get_trackbar_value(trackbar4_MinC, window_name2)
        maxCNT = get_trackbar_value(trackbar3_MaxC, window_name2)

        _, frame = videoStream
        frame = imutils.resize(frame, width=600)
        binary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, thresh1 = cv2.threshold(binary, threshVal, maxThresh, cv2.THRESH_BINARY)

        denoised = denoise(thresh1, noiseVal)

        CNTs = draw_contours(denoised, frame, minCNT, maxCNT)

        cv2.imshow(window_name, frame)
        cv2.imshow(window_name2, denoised)
        if cv2.waitKey(1) & 0xFF == ord('b'):
            cv2.destroyAllWindows()
            break
    print("Got " + str(len(CNTs)) + " contours")
    return CNTs

############################# video BG substractor ###############################3
def substractbg(frame, back_sub):
    cont = []
    # Create kernel for morphological operation
    kernel = np.ones((2, 2), np.uint8)

    # Use every frame to calculate the foreground mask and update BG
    fg_mask = back_sub.apply(frame)

    # Close dark gaps in foreground object using closing
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Remove salt and pepper noise with a median filter
    fg_mask = cv2.medianBlur(fg_mask, 5)  # usar entre 5 y 11?

    # Threshold the image to make it either black or white
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    # fg_mask = cv2.Canny(fg_mask,10,15)

    # Find the index of the largest contour and draw bounding box
    fg_mask_bb = fg_mask
    contours, hierarchy = cv2.findContours(fg_mask_bb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]

    # If there are no countours
    if len(areas) > 1:
        # Find the largest moving object in the image
        max_index = np.argmax(areas)
        # Draw the bounding box
        cont = contours[max_index]

    return cont, fg_mask_bb

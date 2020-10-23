from __future__ import print_function  # Python 2/3 compatibility
import cv2  # Import the OpenCV library
import numpy as np  # Import Numpy library


# recorte de frame
# frametop = frame[:600]
# framebot = frame[600:]

def substractbg(frame, back_sub):
    cont = []
    # Create kernel for morphological operation
    kernel = np.ones((2, 2), np.uint8)

    # Use every frame to calculate the foreground mask and update BG
    fg_mask = back_sub.apply(frame)

    # Close dark gaps in foreground object using closing
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Remove salt and pepper noise with a median filter
    fg_mask = cv2.medianBlur(fg_mask, 5) #usar entre 5 y 11?

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

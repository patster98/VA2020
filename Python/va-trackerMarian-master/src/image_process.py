import cv2


def draw_contours(denoised, original, min_contour_area, max_contour_area):

    result = []
    contours, hierarchy = cv2.findContours(denoised, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])

        if min_contour_area <= area <= max_contour_area:
            result.append(contours[i])
            cv2.drawContours(original, contours[i], -1, (0, 0, 255), 2)
    if len(result) > 0:
        result = result[0:1]
    return result


def denoise(frame, noise_value):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (noise_value, noise_value))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

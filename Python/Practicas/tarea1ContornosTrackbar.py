import cv2
def on_trackbar_change(val):
    # do nothing
    pass
window_name = 'Binary'
window_name2 = 'Denoise contour'

trackbar_name = 'ThreshLim'
trackbar_name2 = 'ThreshVal'
trackbar_name3 = 'Error' #entre 1 y 50
cv2.namedWindow(window_name)
cv2.namedWindow(window_name2)
slider_max = 100
slider3_max = 50
# cv2.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar_change)
# cv2.createTrackbar(trackbar_name2, window_name, 0, 255, on_trackbar_change)
cv2.createTrackbar(trackbar_name3, window_name2, 1, slider3_max, on_trackbar_change)


def contour():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        _, frame = cap.read()
        _, frame2 = cap.read()

        # trackbar_val = cv2.getTrackbarPos(trackbar_name, window_name)
        # trackbar_val2 = cv2.getTrackbarPos(trackbar_name2, window_name)
        trackbar_val3 = cv2.getTrackbarPos(trackbar_name3, window_name2)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # _, thresh1 = cv2.threshold(gray, trackbar_val, trackbar_val2, cv2.THRESH_BINARY)
        _, thresh1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        contours_noise, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_denoise, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cnt = contours_denoise[0]
        max_area = cv2.contourArea(cnt)

        for cont in contours_denoise:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)

        perimeter = cv2.arcLength(cnt, True)
        #epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, trackbar_val3, True)

        cv2.drawContours(frame, contours_noise, -1, (0, 0, 255), 3)
        cv2.drawContours(frame2, [approx], -1, (0, 0, 255), 3)
        cv2.drawContours(frame2, contours_denoise, -1, (0, 255, 255), 3)



        #cv2.imshow("Noise", cv2.flip(frame, 1))
        cv2.imshow(window_name2, cv2.flip(frame2, 1))
        cv2.imshow(window_name, cv2.flip(thresh1, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


contour()

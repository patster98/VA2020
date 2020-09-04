import cv2

def on_trackbar_change(val):
    # do nothing
    pass


window_name = 'Binary'
window_name2 = 'Binary2'

trackbar_name = 'ThreshLim'
trackbar_name2 = 'ThreshVal'
cv2.namedWindow(window_name)
slider_max = 100
cv2.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar_change)
cv2.createTrackbar(trackbar_name2, window_name, 0, 255, on_trackbar_change)


# muestra una imagen binaria normal y controla umbral con trackbar
def binary_with_trackbar():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        _, frame = cap.read()
        trackbar_val = cv2.getTrackbarPos(trackbar_name, window_name)
        trackbar_val2 = cv2.getTrackbarPos(trackbar_name2, window_name)


        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, thresh1 = cv2.threshold(gray, trackbar_val, trackbar_val2, cv2.THRESH_BINARY) #cv2.THRESH_BINARY_INV me hace lo mismo al reves
        cv2.imshow(window_name, cv2.flip(thresh1, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Show some stuff
binary_with_trackbar()

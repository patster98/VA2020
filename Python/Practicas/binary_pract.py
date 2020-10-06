import cv2


# generar una imagen binaria a partir de la imagen de la camara
def binary():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        cv2.imshow("Binary", cv2.flip(thresh, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


binary()




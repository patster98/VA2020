import cv2


def denoise():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, thresh1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("Noise", cv2.flip(thresh1, 1))
        cv2.imshow("Denoised", cv2.flip(closing, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


denoise()

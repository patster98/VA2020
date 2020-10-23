import cv2


def contour():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        _, frame = cap.read()
        _, frame2 = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, thresh1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        contours_noise, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # contours_denoise, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        # cnt = contours_denoise[0]
        # max_area = cv2.contourArea(cnt)
        #
        # #toma el contorno mas grande
        # for cont in contours_denoise:
        #     if cv2.contourArea(cont) > max_area:
        #         cnt = cont
        #         max_area = cv2.contourArea(cont)

        perimeter = cv2.arcLength(cnts, True)
        epsilon = 0.01 * cv2.arcLength(cnts, True)
        approx = cv2.approxPolyDP(cnts, epsilon, True)

        cv2.drawContours(frame, contours_noise, -1, (0, 0, 255), 3)
        cv2.drawContours(frame2, [approx], -1, (0, 0, 255), 3)

        cv2.imshow("Noise", cv2.flip(frame, 1))
        cv2.imshow("Denoise", cv2.flip(frame2, 1))
        # cv2.imshow("Binary", cv2.flip(thresh1, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


contour()
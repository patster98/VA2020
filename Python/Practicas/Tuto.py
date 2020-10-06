import cv2
import imutils

# img = cv2.imread('static/images/gaddi.jpg')
# cv2.imshow('Hello There!' , img)
# cv2.waitKey(0) #mantiene esperando la ventana

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  #abro el canal de la webcam del sist = 0
    while True:
        _, image = cap.read() #agrego el _ porq no me importa ese param
        resized = imutils.resize(image, width=500, height=200)
        image_flip = cv2.flip(resized,1)
        rotated = imutils.rotate(image_flip, 0) #es para rotar, 0 no rota
        cv2.rectangle(rotated,(350, 100),(400, 120),(0, 0, 256), 2)
        cv2.imshow("Webcam", rotated)
        #cv2.imshow("sin espejar", resized)
        if cv2.waitKey(1) & 0xFF == ord('z'): #si apreto z (chequea el codigo ascii) me apaga el programa
            break

#para dibujar sobre la imagen
# cv2.rectangle(img, pt1, pt2, color, thickness)
# cv2.line(img, pt1, pt2, color, thickness)
# cv2.circle(img, center, radius, color, thickness)
# cv2.putText(img, text, pt, font, scale, color, thickness)

from math import sqrt, copysign, log10

import cv2

# Ejercicio 1
# Obtener el contorno de una figura principal, dibujarlo en la imagen
# y dibujar también una marca en el centroide y una circunferencia
# de radio proporcional a la raíz cuadrada de m00


def exercise_one():
    image = cv2.imread('static/images/phone1.png')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("original", image)

    contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    # toma el contorno mas grande
    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # draw the contour and center of the shape on the image
    cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)
    cv2.circle(image, (cX, cY), int(sqrt(M["m00"])), (255, 0, 0), 2)
    cv2.circle(image, (cX, cY), 3, (255, 0, 0), -1)
    cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# Ejercicio 3
# Ídem anterior imprimiendo los invariantes de Hu en lugar de los momentos
# Para obtener la matriz de Hu usen el siguiente metodo:
#     huMoments = cv.HuMoments(cnt)


def exercise_three():
    letter_s_one = cv2.imread('../static/images/letterSOne.png')
    gray1 = cv2.cvtColor(letter_s_one, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv2.threshold(gray1, 127, 255, cv.THRESH_BINARY)
    cv2.imshow("letter_s_one", letter_s_one)

    contours1, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt1 = contours1[1]
    # compute the center of the contour
    cv2.drawContours(letter_s_one, [cnt1], -1, (0, 0, 255), 2)
    cv2.imshow("contorno letter s one", letter_s_one)
    moments1 = cv2.moments(cnt1)
    huMoments1 = cv2.HuMoments(moments1)
    # hu[0] is not comparable in magnitude as hu[6].
    # We can use use a log transform given below to bring them in the same range
    for i in range(0, 7):
        huMoments1[i] = -1 * copysign(1.0, huMoments1[i]) * log10(abs(huMoments1[i]))
        print("letter s1 -> h" + str(i) + " " + str(huMoments1[i]))

    letter_s_two = cv2.imread('../static/images/letterSTwo.png')
    gray2 = cv2.cvtColor(letter_s_two, cv.COLOR_RGB2GRAY)
    ret2, thresh2 = cv2.threshold(gray2, 127, 255, cv.THRESH_BINARY)
    cv2.imshow("letter_s_two", letter_s_two)

    contours2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt2 = contours2[1]
    # compute the center of the contour
    cv2.drawContours(letter_s_two, [cnt2], -1, (0, 0, 255), 2)
    cv2.imshow("contorno letter s two", letter_s_two)
    moments2 = cv2.moments(cnt2)
    huMoments2 = cv2.HuMoments(moments2)
    # hu[0] is not comparable in magnitude as hu[6].
    # We can use use a log transform given below to bring them in the same range
    for i in range(0, 7):
        huMoments2[i] = -1 * copysign(1.0, huMoments2[i]) * log10(abs(huMoments2[i]))
        print("letter s2 -> h" + str(i) + " " + str(huMoments2[i]))

    cv2.waitKey(0)

# Ejercicio 4
# Obtener los invariantes de Hu para una forma, y reconocerla luego en otras imágenes


def exercise_four():
    letter_s = cv2.imread('../static/images/letterSTwo.png')
    gray1 = cv2.cvtColor(letter_s, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv2.threshold(gray1, 127, 255, cv.THRESH_BINARY)
    contours1, hierarchy = cv2.findContours(thresh1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt1 = contours1[1]
    cv2.drawContours(letter_s, [cnt1], -1, (0, 0, 255), 2)
    cv2.imshow("letter_s_one", letter_s)
    moments = cv2.moments(cnt1)
    cv2.waitKey(0)
    huMoments = cv2.HuMoments(moments)
#     now in huMoments1 we have letter s invariants
#     lets use shape match
    alphabet = cv2.imread('../static/images/alphabet.jpg')
    gray2 = cv2.cvtColor(alphabet, cv.COLOR_RGB2GRAY)
    ret2, thresh2 = cv2.threshold(gray2, 127, 255, cv.THRESH_BINARY)
    cv2.imshow("alphabet", alphabet)
    contours_alphabet, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours_alphabet:
        moments_alphabet = cv2.moments(contour)
        huMoments_alphabet = cv2.HuMoments(moments_alphabet)
        if cv2.matchShapes(contour, cnt1, cv.CONTOURS_MATCH_I2, 0) < 0.4:
            cv2.drawContours(alphabet, contour, -1, (0, 0, 255), 2)
            cv2.imshow("contornos", alphabet)
            cv2.waitKey(0)

if __name__ == '__main__':

    exercise_one()
    #exercise_three()
    #exercise_four()
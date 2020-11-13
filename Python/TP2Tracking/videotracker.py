import cv2
import sys
import numpy as np
import imutils
from utils.bgSubstract import contDetect

############### Tracker Types #####################

# argtrack = cv2.TrackerBoosting_create()
# argtrack = cv2.TrackerMIL_create()
# argtrack = cv2.TrackerKCF_create()  # muy rapido pero se pierde con oclusion y devuelve siempre true aunque pierda
# argtrack = cv2.TrackerTLD_create() #el mejor para safar los arboles pero muy lento
# argtrack = cv2.TrackerMedianFlow_create()
argtrack = cv2.TrackerCSRT_create()
# argtrack = cv2.TrackerMOSSE_create()

########################################################

tracker = cv2.MultiTracker_create()

start = False
# Create the background subtractor object
# Use the last 700 video frames to build the background
back_sub = cv2.createBackgroundSubtractorMOG2(history=700,
                                              varThreshold=25, detectShadows=True)

videoPath = '/Users/Skyguy/Desktop/Ingenieria/SegundoCuatrimestre/VisionArtificial/VA2020/Python/static/videos' \
            '/CarreraJuegoAutos.mp4'

cap = cv2.VideoCapture(videoPath)

for i in range(0,100):
    _,rld = cap.read()
    cnt, bwimg = contDetect(rld, back_sub)
    if cnt != []:
        car = cnt
cap.release()
print(car)

cap = cv2.VideoCapture(videoPath)
success ,frame_n = cap.read()

while True:
    # aca meto selección de contorno por trackbar y paso como argumento a la funcion de abajo
    # cnt, bwimg = substractbg(frame_n, back_sub)
    cv2.imshow("CNTs", bwimg)
    cv2.imshow("CNT Selector", frame_n)
    if cv2.waitKey(0) & 0xFF == ord('b'):
        break

x, y, w, h = cv2.boundingRect(car)
box2 = (x,y,w,h)
tracker.add(argtrack, frame_n, box2)

if not success:
    print('Video not found')
    print("success:", success, sep=" ")
    sys.exit(1)

while True:

    timer = cv2.getTickCount()
    ret, frame = cap.read()
    drawframe = frame

    if frame is None:
        break

    # frame = imutils.resize(frame, width=600)
    # else:
    #     cv2.putText(img, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.rectangle(drawframe, (15, 15), (200, 90), (255, 0, 255), 2)
    cv2.putText(drawframe, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);
    cv2.putText(drawframe, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    if fps > 60:
        myColor = (20, 230, 20)
    elif fps > 20:
        myColor = (230, 20, 20)
    else:
        myColor = (20, 20, 230)
    cv2.putText(drawframe, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2);

    cv2.imshow("Black and white", bwimg)

    (success, bbox) = tracker.update(frame)
    # print("success:", success, sep=" ")

    for box in bbox:
        (x, y, w, h) = [int(v) for v in box]
        # (x, y, w, h) = [bbox[0,0], bbox[0,1], bbox[0,2], bbox[0,3]]
        rect = cv2.rectangle(drawframe, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(drawframe, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Draw circle in the center of the bounding box
        xc = x + int(w / 2)
        yc = y + int(h / 2)
        cv2.circle(drawframe, (xc, yc), 4, (0, 255, 0), -1)

    cv2.imshow("Tracking", drawframe)
    key = cv2.waitKey(50) & 0xFF

    if key == ord('s'):
        box = cv2.selectROI("Tracking", frame, fromCenter=False,
                            showCrosshair=True)
        # tracker1 = argtrack
        tracker.add(argtrack, frame, box)
        # print(box)

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

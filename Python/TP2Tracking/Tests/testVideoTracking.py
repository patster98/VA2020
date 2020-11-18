import cv2
import sys
import numpy as np
import imutils
from utils.bgSubstract import substractbg

############### Tracker Types #####################

# argtrack = cv2.TrackerBoosting_create()
# argtrack = cv2.TrackerMIL_create()
argtrack = cv2.TrackerKCF_create()  # muy rapido pero se pierde con oclusion y devuelve siempre true aunque pierda
# argtrack = cv2.TrackerTLD_create() #el mejor para safar los arboles pero muy lento
# argtrack = cv2.TrackerMedianFlow_create()
# argtrack = cv2.TrackerCSRT_create()
# argtrack = cv2.TrackerMOSSE_create()

########################################################

tracker = cv2.MultiTracker_create()

# Create the background subtractor object
# Use the last 700 video frames to build the background
back_sub = cv2.createBackgroundSubtractorMOG2(history=700,
                                              varThreshold=25, detectShadows=True)

videoPath = '/Users/Skyguy/Desktop/Ingenieria/SegundoCuatrimestre/VisionArtificial/VA2020/Python/static/videos' \
            '/CarreraJuegoAutos.mp4'

cap = cv2.VideoCapture(videoPath)
success,_ = cap.read()

cnt = []
box2 = ()
x = 0
y = 0
x2 = 0
y2 = 0
start = False

if not success:
    print('Video not found')
    print("success:", success, sep=" ")
    sys.exit(1)

success = False
while True:

    timer = cv2.getTickCount()
    ret, frame = cap.read()
    drawframe = frame
    _, img2 = cap.read()

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

    # cnt.append(cnt)

    if start: #and not success:
        cnt, bwimg = substractbg(img2, back_sub)
        if cnt != []:
            x, y, w, h = cv2.boundingRect(cnt)
            # print(x,y,w,h)
            box2 = (x,y,w,h)

            # comparar con el ultimo box y si no esta cerca no lo agrego
            compare = ((x - x2)**2 + (y-y2)**2)**0.5
            # print("compare:", compare, sep=" ")sacar

            if float(compare) < 400:
                tracker.add(argtrack, frame, box2)
            else:
                frame = img2.copy()
                frame = imutils.resize(frame, width=600)
                bbox = ()
                success = False
        cnt = []
        cv2.imshow("Black and white", bwimg)

    (success, bbox) = tracker.update(frame)
    print("success:", success, sep=" ")

    for box in bbox:
        (x, y, w, h) = [int(v) for v in box]
        # (x, y, w, h) = [bbox[0,0], bbox[0,1], bbox[0,2], bbox[0,3]]
        rect = cv2.rectangle(drawframe, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(drawframe, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Draw circle in the center of the bounding box
        xc = x + int(w / 2)
        yc = y + int(h / 2)
        cv2.circle(drawframe, (xc, yc), 4, (0, 255, 0), -1)


    x2 = x
    y2 = y
    cv2.imshow("Tracking", drawframe)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('b'):
        start = True

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

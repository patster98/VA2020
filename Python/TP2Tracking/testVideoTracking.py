import cv2
import sys
import numpy as np
import imutils
from pathlib import Path
from utils.bgSubstract import substractbg

############### Tracker Types #####################

# argtrack = cv2.TrackerBoosting_create()
# argtrack = cv2.TrackerMIL_create()
argtrack = cv2.TrackerKCF_create()  # muy rapido pero se pierde con oclusion
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
            '/carsRt9_3.avi'
# videoPath = 0
# videoPath = str(Path('static/videos/race.mp4').resolve(strict=False))

cap = cv2.VideoCapture(videoPath)
success, frame = cap.read()
cnt = []
box2 = []
start = None

if not success:
    print('Video not found')
    print(success)
    sys.exit(1)

# bbox = cv2.selectROI("Tracking", frame, False)
# tracker.init(frame, bbox)


# def drawBox(img, bbox):
#     x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
#     cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3)
#     cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:

    timer = cv2.getTickCount()
    ret, img = cap.read()
    _, img2 = cap.read()

    if img is None:
        break

    # img = imutils.resize(img, width=600) por alguna razon esto no anda
    (success, bbox) = tracker.update(img)

    # if success:
    # drawBox(img, bbox)

    for box in bbox:
    # print(bbox)
    # if bbox != ():
        (x, y, w, h) = [int(v) for v in box]
        # (x, y, w, h) = [bbox[0,0], bbox[0,1], bbox[0,2], bbox[0,3]]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Tracking", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Draw circle in the center of the bounding box
        x2 = x + int(w / 2)
        y2 = y + int(h / 2)
        cv2.circle(img, (x2, y2), 4, (0, 255, 0), -1)
    # else:
    #     cv2.putText(img, "Lost", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.rectangle(img, (15, 15), (200, 90), (255, 0, 255), 2)
    cv2.putText(img, "Fps:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);
    cv2.putText(img, "Status:", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2);

    # if success is False:
    #     # bbox = []
    #     bbox = cv2.selectROI("Tracking", img, False)
    #     print()

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    if fps > 60:
        myColor = (20, 230, 20)
    elif fps > 20:
        myColor = (230, 20, 20)
    else:
        myColor = (20, 20, 230)
    cv2.putText(img, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2);

    cv2.imshow("Tracking", img)
    key = cv2.waitKey(1) & 0xFF

    # cnt.append(cnt)
    cnt, bwimg = substractbg(img2, back_sub)
    if cnt != [] and start:
        # print(cnt)
        if box2 != []:
            box2 = []

        x, y, w, h = cv2.boundingRect(cnt)
        # print(x,y,w,h)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),

        box2.append(x)
        box2.append(y)
        box2.append(w)
        box2.append(h)



        # Print the centroid coordinates (we'll use the center of the
        # # bounding box) on the image
        # text = "x: " + str(x2) + ", y: " + str(y2)
        # cv2.putText(frame, text, (x2 - 10, y2 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tracker.add(argtrack, img, tuple(box2))
        cv2.imshow("Black and white", bwimg)

    if key == ord('b'):
        start = True

    if key == ord('s'):
        box = cv2.selectROI("Tracking", img, fromCenter=False,
                            showCrosshair=True)
        # tracker1 = argtrack
        tracker.add(argtrack, img, box)
        # print(box)

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

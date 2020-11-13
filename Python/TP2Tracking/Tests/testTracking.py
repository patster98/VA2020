import cv2
import numpy as np

# stream = "static/videos/carsRt9_3.avi"
tracker = cv2.TrackerKCF_create()
video = cv2.VideoCapture(0)
# video = cv2.VideoCapture(stream)
while True:
    k,frame = video.read()
    cv2.imshow("Tracking",cv2.flip(frame, 1))
    k = cv2.waitKey(30) & 0xff
    if k == ord('s'):
        break
bbox = cv2.selectROI(frame, False)

ok = tracker.init(frame, bbox)
cv2.destroyWindow("ROI selector")

while True:
    ok, frame = video.read()
    ok, bbox = tracker.update(frame)

    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]),
              int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0,0,255), 2, 2)

    cv2.imshow("Tracking", cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

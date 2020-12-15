from __future__ import print_function  # Python 2/3 compatibility
import cv2
import sys
import numpy as np
import imutils
from utils.bgSubstract import cntSelect
from utils.operations import lapNumber, speedCalc, trackerCreation

############### Tracker Types #####################
# tracker = cv2.TrackerBoosting_create()
# tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerKCF_create()  # muy rapido pero se pierde con oclusion y devuelve siempre true aunque pierda
# tracker = cv2.TrackerTLD_create() #el mejor para safar los arboles pero muy lento
# tracker = cv2.TrackerMedianFlow_create()
tracker = cv2.TrackerCSRT_create()
# tracker = cv2.TrackerMOSSE_create()

################### Main Code ########################
src = "C://Users/Skyguy/Desktop/Ingenieria/SegundoCuatrimestre/VisionArtificial/VA2020/Python/static/videos/CarreraJuegoAutos.mp4"
def main():
    cap = cv2.VideoCapture(src)

    trackerMode = cv2.MultiTracker_create()
    video = cap.read()
    gotVideo, frame = video

    if not gotVideo:
        print('Video not found')
        print("success:", gotVideo, sep=" ")
        sys.exit(1)
    else:
        print("Select CNT to track")

    videoFrame = imutils.resize(frame, width=600)

    ############################### Operations on first frame ##############################
    contours = cntSelect(video)
    trackerCreation(trackerMode, tracker, videoFrame, contours)

    # constants
    fps = cap.get(cv2.CAP_PROP_FPS)
    finish_line_x1 = 310
    finish_line_x2 = 330
    finish_line_y1 = 12
    finish_line_y2 = 108
    carLength_meters = 5.12  # Car std measurement from Indianapolis 500 car series

    # variables
    carLength_pixels = None
    prev_x = 0
    prev_y = 0
    total_distance = 0
    video_time = 0
    lap_times = [0]
    currentSpeed = 0
    displayedSpeed = 0
    frameCount = 0
    current_lapCounter = 1

    (success, boxes) = trackerMode.update(videoFrame)
    ################# get first value for comparisson ##########
    for box in boxes:
        (x, y, w, h) = [int(b) for b in box]
        prev_x = x
        prev_y = y
        carLength_pixels = w

    ###################### Main loop ############################
    while True:
        succ, frame = cap.read()

        if succ:
            videoFrame = imutils.resize(frame, width=600)
        else:
            frametop = lapFrame[0:120, 250:400]
            cv2.imwrite("C://Users/Skyguy/Desktop/Ingenieria/SegundoCuatrimestre/VisionArtificial/VA2020/Python"
                        "/TP2Tracking/Winner/winnerCar.jpg",frametop)
            sys.exit(1)

        cv2.line(videoFrame, (finish_line_x1, finish_line_y1), (finish_line_x1, finish_line_y2), (0, 255, 153), thickness= 2)
        cv2.line(videoFrame, (finish_line_x2, finish_line_y1), (finish_line_x2, finish_line_y2), (0, 255, 153), thickness= 2)
        if fps > 60:
            myColor = (20, 230, 20)
        elif fps > 20:
            myColor = (230, 20, 20)
        else:
            myColor = (20, 20, 230)
        cv2.putText(videoFrame,"FPS: "+ str(int(fps)), (400, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2);

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

        (success, boxes) = trackerMode.update(videoFrame)
        for box in boxes:
            (x, y, w, h) = [int(b) for b in box]
            cv2.rectangle(videoFrame, (x, y), (x + w, y + h), (0, 143, 255), 2, cv2.LINE_AA)
            # Draw circle in the center of the bounding box
            xc = x + int(w / 2)
            yc = y + int(h / 2)
            cv2.circle(videoFrame, (xc, yc), 3, (0, 143, 255), -1)
            cv2.putText(videoFrame, "winning car",
                        (x-15, y-5),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.375,
                        (0, 143, 255),
                        1,
                        cv2.LINE_AA)

            current_lapCounter = lapNumber(current_lapCounter,
                                            lap_times,
                                            x, y, w, h,
                                            finish_line_x1,
                                            finish_line_x2,
                                            finish_line_y1,
                                            finish_line_y2,
                                            video_time)
            if current_lapCounter == 2:
                lapFrame = videoFrame

            speed, new_total_distance = speedCalc(
                total_distance,
                prev_x,
                prev_y,
                x, y,
                fps,
                video_time,
                carLength_pixels,
                carLength_meters, currentSpeed)

            currentSpeed = speed
            total_distance = new_total_distance
            prev_x = x
            prev_y = y

        cv2.putText(videoFrame, "Lap count: " + str(current_lapCounter),
                    (30, 290), cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA)

        cv2.putText(videoFrame, "Distance covered: " + str(total_distance) + " m",
                    (30, 330),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA)

        if frameCount == 5:
            #display speed every 5 frames
            displayedSpeed = currentSpeed
            frameCount = 0
        else:
            frameCount = frameCount + 1

        cv2.putText(videoFrame, "Speed: " + str(displayedSpeed) + " Km/h",
                    (30, 310),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA)

        for time_index in range(1, len(lap_times)):
            aux = time_index + 1
            cv2.putText(videoFrame, "Lap " + str(time_index) + ": " + str(np.floor(lap_times[time_index] / 1000)) + " secs",
                        (20, int(50 * (aux * 0.3))),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        (140, 20, 74),
                        1,
                        cv2.LINE_AA)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.waitKey(-1)

        video_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        cv2.imshow("Video", videoFrame)

main()
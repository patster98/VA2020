import cv2
from src.trackbar import create_trackbar, get_trackbar_value
from src.image_process import denoise, draw_contours
import argparse
import imutils
import math


def create_trackers(trackers, tracker, frame, contours):
    for contour in contours:
        trackers.add(tracker, frame, cv2.boundingRect(contour))


def get_contours(video):
    window_name = "Trackbars"
    cv2.namedWindow(window_name)
    trackbar_thresh_value_name = "Threshold Value"
    trackbar_noise_val_name = "Noise Value"
    trackbar_contour_area_min_name = "Contour Area Min"
    trackbar_contour_area_max_name = "Contour Area Max"
    trackbar_compare_value_name = "Compare Value"
    threshold_max = 255
    noise_max = 20
    contour_area_max = 500
    contour_area_min = 500
    create_trackbar(trackbar_thresh_value_name, window_name, threshold_max, 180)
    create_trackbar(trackbar_noise_val_name, window_name, noise_max)
    create_trackbar(trackbar_contour_area_min_name, window_name, contour_area_min, 450)
    create_trackbar(trackbar_contour_area_max_name, window_name, contour_area_max, 500)
    create_trackbar(trackbar_compare_value_name, window_name, 200)

    while True:
        threshold_value = get_trackbar_value(trackbar_name=trackbar_thresh_value_name, window_name=window_name)
        noise_value = get_trackbar_value(trackbar_name=trackbar_noise_val_name, window_name=window_name)
        contour_area_min = get_trackbar_value(trackbar_name=trackbar_contour_area_min_name, window_name=window_name)
        contour_area_max = get_trackbar_value(trackbar_name=trackbar_contour_area_max_name, window_name=window_name)

        _, frame = video
        frame = imutils.resize(frame, width=600)
        binary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret1, thresh1 = cv2.threshold(binary, threshold_value, threshold_max, cv2.THRESH_BINARY)

        denoised = denoise(thresh1, noise_value)

        contours = draw_contours(denoised, frame, contour_area_min, contour_area_max)

        cv2.imshow("Original", frame)
        cv2.imshow("Denoised", denoised)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("Got " + str(len(contours)) + " contours")
    return contours


def calculate_velocity_and_total_distance(total_distance, previous_x, previous_y, x, y, fps,video_time, car_width_in_pixels, car_width_in_meters):
    x_part = math.pow(x - previous_x, 2)
    y_part = math.pow(y - previous_y, 2)
    # distance = ( (x2 - x1)^2 + (y2 - y1)^2 )^0.5

    current_distance_in_pixels = math.floor(math.sqrt(x_part + y_part))
    current_distance_in_meters = (current_distance_in_pixels * car_width_in_meters) / car_width_in_pixels

    distance_in_meters_per_seconds = (current_distance_in_meters * fps) / 1 if video_time != 0 else video_time

    distance_in_kilometer_per_hour = distance_in_meters_per_seconds * 3.6
    new_total_distance = total_distance + current_distance_in_meters
    return math.floor(distance_in_kilometer_per_hour), math.floor(new_total_distance)


def check_lap(current_lap_counter, lap_times, x, y, w, h, finish_line_x1, finish_line_x2, finish_line_y1, finish_line_y2,
              video_time):
    if finish_line_x1 <= x + w < finish_line_x2 and\
            finish_line_y1 <= y + h <= finish_line_y2\
            and (video_time - lap_times[-1] > 1500):
        lap_times.append(video_time)
        return current_lap_counter + 1
    else:
        return current_lap_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
                    default="C://Users/marianol/projects/austral/va-tracker/resources/video-nuevo.mp4",
                    help="path to input video file")
    args = vars(ap.parse_args())
    vs = cv2.VideoCapture(args["video"])

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    trackers = cv2.MultiTracker_create()

    video = vs.read()
    _, frame = video

    show_video_frame = imutils.resize(frame, width=600)
    contours = get_contours(video)
    create_trackers(trackers, OPENCV_OBJECT_TRACKERS["csrt"](), show_video_frame, contours)

    # constants
    fps = vs.get(cv2.CAP_PROP_FPS)
    finish_line_x1 = 310
    finish_line_x2 = 330
    finish_line_y1 = 12
    finish_line_y2 = 108
    car_width_in_meters = 4  # Medida standard de largo de autos

    # variables
    car_width_in_pixels = None
    previous_x = 0
    previous_y = 0
    total_distance = 0
    video_time = 0
    lap_times = [0]
    current_velocity = 0
    display_velocity = 0
    frame_counter = 0
    current_lap_counter = 1

    (success, boxes) = trackers.update(show_video_frame)

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        previous_x = x
        previous_y = y
        car_width_in_pixels = w

    while True:
        video = vs.read()

        _, frame = video

        show_video_frame = imutils.resize(frame, width=600)

        cv2.line(show_video_frame, (finish_line_x1, finish_line_y1), (finish_line_x1, finish_line_y2), (255, 0, 0))
        cv2.line(show_video_frame, (finish_line_x2, finish_line_y1), (finish_line_x2, finish_line_y2), (255, 0, 0))
        if cv2.waitKey(1) & 0xFF == ord('f'):
            break

        (success, boxes) = trackers.update(show_video_frame)

        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(show_video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            current_lap_counter = check_lap(current_lap_counter,
                                            lap_times,
                                            x, y, w, h,
                                            finish_line_x1,
                                            finish_line_x2,
                                            finish_line_y1,
                                            finish_line_y2,
                                            video_time)

            velocity, new_total_distance = calculate_velocity_and_total_distance(
                total_distance,
                previous_x,
                previous_y,
                x, y,
                fps,
                video_time,
                car_width_in_pixels,
                car_width_in_meters)

            current_velocity = velocity
            total_distance = new_total_distance
            previous_x = x
            previous_y = y

        cv2.putText(show_video_frame, "Current lap: " + str(current_lap_counter),
                    (450, 30), cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA)

        cv2.putText(show_video_frame, "Total meters: " + str(total_distance),
                    (30, 330),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA)

        if frame_counter == 5:
            display_velocity = current_velocity
            frame_counter = 0
        else:
            frame_counter = frame_counter + 1

        cv2.putText(show_video_frame, "Km/h: " + str(display_velocity),
                    (30, 300),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA)

        for time_index in range(1, len(lap_times)):
            aux = time_index + 1
            cv2.putText(show_video_frame, "Lap " + str(time_index) + ": " + str(math.floor(lap_times[time_index] / 1000)) + " secs",
                        (20, int(50 * (aux * 0.3))),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.waitKey(-1)

        video_time = vs.get(cv2.CAP_PROP_POS_MSEC)
        cv2.imshow("Video", show_video_frame)


main()

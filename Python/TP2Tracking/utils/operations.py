import cv2
import numpy as np
############################## Create and update trackers #############
def trackerCreation(trackerMode, tracker, frame, contours):
    for contour in contours:
        trackerMode.add(tracker, frame, cv2.boundingRect(contour))


############################## SPEED + POSITION ######################
def speedCalc(total_distance, previous_x, previous_y, x, y, fps, video_time, car_length_in_pixels, car_length_in_meters, currentSpeed):
    last_xy = np.array([previous_x, previous_y])
    new_xy = np.array([x, y])
    mat_z = new_xy - last_xy
    euclidDist = np.sqrt(np.einsum('i,i->', mat_z, mat_z))
    # distance = ( (x2 - x1)^2 + (y2 - y1)^2 )^0.5

    current_distance_in_pixels = np.floor(euclidDist)
    current_distance_in_meters = (current_distance_in_pixels * car_length_in_meters) / car_length_in_pixels

    speed_in_meters_per_seconds = (current_distance_in_meters * fps) / 1 if video_time != 0 else video_time

    speed_in_kilometers_per_hour = speed_in_meters_per_seconds * 3.6

    #avoid sudden changes in speed due to fall in fps
    if abs(currentSpeed-speed_in_kilometers_per_hour)>20:
        finalSpeed = (speed_in_kilometers_per_hour + currentSpeed) / 2
    else:
        finalSpeed = currentSpeed

    new_total_distance = total_distance + current_distance_in_meters
    return np.floor(finalSpeed), np.floor(new_total_distance)

################################ LAP COUNTER ###########################
def lapNumber(current_lap_counter, lap_times, x, y, w, h, finish_line_x1, finish_line_x2, finish_line_y1, finish_line_y2,
              video_time):
    if finish_line_x1 <= x + w < finish_line_x2 and\
            finish_line_y1 <= y + h <= finish_line_y2\
            and (video_time - lap_times[-1] > 1500):
        lap_times.append(video_time)
        return (current_lap_counter + 1)
    else:
        return current_lap_counter

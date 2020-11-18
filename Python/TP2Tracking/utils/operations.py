import cv2
import  math
############################## Create and update trackers #############
def trackerCreation(trackerMode, tracker, frame, contours):
    for contour in contours:
        trackerMode.add(tracker, frame, cv2.boundingRect(contour))


############################## SPEED + POSITION ######################
def speedCalc(total_distance, previous_x, previous_y, x, y, fps,video_time, car_width_in_pixels, car_width_in_meters):
    x_part = math.pow(x - previous_x, 2)
    y_part = math.pow(y - previous_y, 2)
    # distance = ( (x2 - x1)^2 + (y2 - y1)^2 )^0.5

    current_distance_in_pixels = math.floor(math.sqrt(x_part + y_part))
    current_distance_in_meters = (current_distance_in_pixels * car_width_in_meters) / car_width_in_pixels

    velocity_in_meters_per_seconds = (current_distance_in_meters * fps) / 1 if video_time != 0 else video_time

    velocity_in_kilometer_per_hour = velocity_in_meters_per_seconds * 3.6
    new_total_distance = total_distance + current_distance_in_meters
    return math.floor(velocity_in_kilometer_per_hour), math.floor(new_total_distance)

################################ LAP COUNTER ###########################
def lapNumber(current_lap_counter, lap_times, x, y, w, h, finish_line_x1, finish_line_x2, finish_line_y1, finish_line_y2,
              video_time):
    if finish_line_x1 <= x + w < finish_line_x2 and\
            finish_line_y1 <= y + h <= finish_line_y2\
            and (video_time - lap_times[-1] > 1500):
        lap_times.append(video_time)
        return current_lap_counter + 1
    else:
        return current_lap_counter

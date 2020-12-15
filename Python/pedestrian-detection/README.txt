Python people on store detector based on SVM and HOG.
To run in console:
    python detect.py -c <video or image> -i <image path if image was chosen> -v <video path if video was chosen>
    example:
        python detect.py -c video -v images/OpticaValen_Trim.mp4


Esto usabamos para chequear si el auto pasaba la linea de meta en el tracker, por si les sirve:
def lapNumber(current_lap_counter, lap_times, x, y, w, h, finish_line_x1, finish_line_x2, finish_line_y1, finish_line_y2,
              video_time):
    if finish_line_x1 <= x + w < finish_line_x2 and\
            finish_line_y1 <= y + h <= finish_line_y2\
            and (video_time - lap_times[-1] > 1500):
        lap_times.append(video_time)
        return (current_lap_counter + 1)
    else:
        return current_lap_counter
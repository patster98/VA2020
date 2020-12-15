# USAGE
# python detect.py --images images
#python detect.py -c video -v images/OpticaValen_Trim.mp4

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False, help="path to images directory", default="images")
ap.add_argument("-v", "--video", required=False, help="path to video", default="images/univ.avi")
ap.add_argument("-c", "--capture", required=True, help="capture video or image", default= "image")


args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# constants
p1_line = (180, 85)
p2_line = (325, 115)

# variables
peopleCount = 0
lap_times = [0]
video_time = 0

if args["capture"] == "image":
	# loop over the image paths
	imagePaths = list(paths.list_images(args["images"]))
	for imagePath in imagePaths:
		# load the image and resize it to (1) reduce detection time
		# and (2) improve detection accuracy
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=min(400, image.shape[1]))
		orig = image.copy()

		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
			padding=(8, 8), scale=1.05)

		# draw the original bounding boxes
		for (x, y, w, h) in rects:
			cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

		# show some information on the number of bounding boxes
		filename = imagePath[imagePath.rfind("/") + 1:]
		print("[INFO] {}: {} original boxes, {} after suppression".format(
			filename, len(rects), len(pick)))

		# show the output images
		cv2.imshow("Before NMS", orig)
		cv2.imshow("After NMS", image)
		cv2.waitKey(0)

elif args["capture"]=="video":
	cap = cv2.VideoCapture(args["video"])
	while True:
		# grab the current frame, then handle if we are using a
		# VideoStream or VideoCapture object
		frame = cap.read()
		timer = cv2.getTickCount()
		frame = frame[1] if args.get("video", False) else frame
		# check to see if we have reached the end of the stream
		if frame is None:
			break
		# resize the frame (so we can process it faster)
		frame = imutils.resize(frame, width=min(400, frame.shape[1]))
		# detect people in the image
		(rects, weights) = hog.detectMultiScale(frame, winStride=(6, 6),
												padding=(8, 8), scale=1.05)


		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

		# draw the final bounding boxes
		# xB = x+w , yB = y+h
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
			if (yA + (yB-yA)/2) > p1_line[1] and (video_time - lap_times[-1] )> 1500:
				peopleCount += 1
				lap_times.append(video_time)



		# add fps count
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
		cv2.putText(frame, "FPS: " + str(int(fps)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 230, 40), 2)

		# draw line to count people inside the store
		cv2.line(frame,  p1_line, p2_line, (0, 255, 153),
				 thickness=2)

		# add people counter sign
		cv2.putText(frame, "People inside: " + str(int(peopleCount)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 230, 40), 2)
		if peopleCount > 2:
			cv2.putText(frame, "LIMIT EXCEEDED!", (80, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
						(0, 0, 255), 2)
			cv2.putText(frame, "Leave out: " + str(int(peopleCount - 2)) + " customers", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
						(0, 0, 255), 2)

		# show the output images
		video_time = cap.get(cv2.CAP_PROP_POS_MSEC)
		cv2.imshow("After NMS", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			cap.release()
			break

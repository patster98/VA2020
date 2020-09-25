import cv2
from utils.hu_moments_generation import generate_hu_moments_file
from utils.testing_model import load_and_test
from utils.training_model import train_model
from utils.video_contours import createTrackbars

img = int(input("Select 1 for img ML, 0 for video: "))
generate_hu_moments_file()
model = train_model()
if img != 1:
    createTrackbars()
    cap = cv2.VideoCapture(0)
    load_and_test(model, cap, img=0)
else:
    load_and_test(model, 0, img=1)
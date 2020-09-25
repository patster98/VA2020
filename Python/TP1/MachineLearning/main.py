import cv2
from utils.hu_moments_generation import generate_hu_moments_file
from utils.testing_model import load_and_testIMG, load_and_testVID
from utils.training_model import train_model
from utils.video_contours import createTrackbars

img = int(input("Select 1 for img ML, 0 for cam: "))
generate_hu_moments_file()
model = train_model()
if img != 1:
    createTrackbars()
    cap = cv2.VideoCapture(0)
    load_and_testVID(model, cap)
else:
    load_and_testIMG(model)
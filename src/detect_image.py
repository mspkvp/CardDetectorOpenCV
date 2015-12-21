import card_detector as detector
import cv2
import sys

#load the database
detector.load_cards()

#read image from argument
image = cv2.imread(sys.argv[1])

detector.detect_cards(image)
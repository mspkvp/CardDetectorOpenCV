import cv2
import sys
import cv2.xfeatures2d
import numpy as np
import store_keypoints as skp
import cPickle as pickle
#contours documentation: http://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html#gsc.tab=0

SIFT_DETECTOR = cv2.xfeatures2d.SIFT_create()
#read the parameter image
image = cv2.imread(sys.argv[1])

#obtain a grayscale version
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#apply a threshold to the grayscale version
ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#obtain the contours
#since we're supposing the card is in a well contrasted background with not many other objects
#we can just retrieve the external contour (RETR_EXTERNAL)
image2, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#select the largest contour we found
contour = contours[0]
for c in contours[1:]:
	if len(c) > len(contour):
		contour = c

#draw the contours (just to check the result)
#cv2.drawContours(image, contours, -1, (0,255,0), 3)
#cv2.imshow("contours", image)
#cv2.waitKey(0)


#calculate the perimeter of the countour
#this will be used to determine the precision of approxPolyDP
peri = cv2.arcLength(contour, True)

#obtain the corners of the quadriletral that approximates the contour
approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
approx = np.array(approx, np.float32)

#create a rectangle to which the contour will be transformed
height = 400
width = 300
tl = [0,0]
bl = [0,height - 1]
br = [width - 1,height - 1]
tr = [width - 1,0]
h = np.array([ tl, bl, br, tr ], np.float32)

#determine the transformation and warp the original image
transform = cv2.getPerspectiveTransform(approx, h)
warp = cv2.warpPerspective(image, transform, (width,height))

#determination of the feature points
#grayscale and blur
gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
#run SIFT
keypoints, descriptors = SIFT_DETECTOR.detectAndCompute(gray, None)

#save keypoints and descriptors in file
pickle.dump(skp.pickle_keypoints(keypoints, descriptors), open("database/" + sys.argv[2] + ".sift", "wb+"))
#save image
cv2.imwrite("database/" + sys.argv[2] + ".png", warp)

#draw it with the key points
warp = cv2.drawKeypoints(gray, keypoints, warp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Training image", warp)

cv2.waitKey(0)
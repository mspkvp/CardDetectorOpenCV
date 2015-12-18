import cv2
import cv2.xfeatures2d
import numpy as np
import sys
import store_keypoints as skp
import cPickle as pickle
import os.path

card_values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "queen", "jack", "king", "ace"]
card_suits = ["clubs", "diamonds", "hearts", "spades"]
card_images = dict()
card_features = dict()

MIN_MATCH_COUNT = 0
# Initiate SIFT detector
SIFT_DETECTOR = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10, 1.6)


def load_cards():
    #load the descriptors and images for each card
    for suit in card_suits:
        for value in card_values:
            card_id = value + "_" + suit
            #check if we have the card trained
            if(os.path.isfile("database/" + card_id + ".sift") and os.path.isfile("database/" + card_id + ".sift")):
                #load the image
                card_images[card_id] = cv2.imread("database/" + card_id + ".png")
                #load the features
                card_features[card_id] = skp.unpickle_keypoints(pickle.load(open("database/" + card_id + ".sift")))


def detect_card_in_image(acquired_image, acquired_image_kp, acquired_image_des, card_id):
    #check if card is trained
    if card_id in card_features:
        img1 = card_images[card_id]
        img2 = cv2.cvtColor(acquired_image, cv2.COLOR_BGR2GRAY)

        kp1, des1 = card_features[card_id]
        kp2 = acquired_image_kp
        des2 = acquired_image_des

        #perform matching
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        #homography calculation
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            return 0, dst #return 0 for confirmation and contour of the card
        else:
            return -1, None #card not found
    else:
        return -2, None #card not in database


#load the database
load_cards()

#read image from argument
image = cv2.imread(sys.argv[1])
#grayscale it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blur it
blur = cv2.GaussianBlur(gray, (5,5), 0)
#detect features
kp, des = SIFT_DETECTOR.detectAndCompute(blur,None)

#this is is just to display the image
tmp = cv2.imread(sys.argv[1])
cv2.drawKeypoints(tmp, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("features", tmp)

ret, contour = detect_card_in_image(image, kp, des, sys.argv[2])

if ret == 0:
    #card found, display it with the contour
    cv2.imshow("final", cv2.polylines(image,[np.int32(contour)],True,255,3, cv2.LINE_AA))
    print "card found"
    cv2.waitKey(0)
elif ret == -1:
    print "card not found"
    cv2.waitKey(0)
elif ret == -2:
    print "card not in database"
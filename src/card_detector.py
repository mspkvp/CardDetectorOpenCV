import cv2
import cv2.xfeatures2d
import numpy as np
import sys
import os.path

card_values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "queen", "jack", "king", "ace"]
card_suits = ["clubs", "diamonds", "spades", "hearts"]
card_images = dict()
card_features = dict()

MIN_MATCH_COUNT = 4
fontFace = cv2.FONT_HERSHEY_PLAIN;
fontScale = 2;
thickness = 3;
winnerText = "WINNER"
winnerTextSize = cv2.getTextSize(winnerText, fontFace, fontScale, thickness)[0]
loserText = "LOSER"
loserTextSize = cv2.getTextSize(winnerText, fontFace, fontScale, thickness)[0]

# Initiate SIFT detector
SIFT_DETECTOR = cv2.xfeatures2d.SIFT_create()

def compare_cards(card_ids):
    winning_id = card_ids[0]
    winning_card = winning_id.split("_")
    for card_id in card_ids[1:]:
        new_card = card_id.split("_")

        n = card_values.index(new_card[0]) - card_values.index(winning_card[0]) #compares value

        if(n == 0): #compares suits
            n = card_values.index(new_card[2]) - card_values.index(winning_card[2])

        if (n > 0):
            winning_id = card_id
            winning_card = new_card

    return winning_id


def load_cards():
    #load the descriptors and images for each card
    for suit in card_suits:
        for value in card_values:
            card_id = value + "_" + suit
            #check if we have the card trained
            if(os.path.isfile("database/" + card_id + ".sift")):
                #load the image
                card_images[card_id] = cv2.imread("database/" + card_id + ".png")
                #calculate the features
                gray = cv2.imread("database/" + card_id + ".png", 0)
                card_features[card_id] = SIFT_DETECTOR.detectAndCompute(gray, None)


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
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
        search_params = dict(checks = 100)

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

            h, w, a = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            return len(good), dst #return number of matches for confirmation and contour of the card
        else:
            return -1, None #card not found
    else:
        return -2, None #card not in database


def detect_cards(image):
    #grayscale the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #detect features
    kp, des = SIFT_DETECTOR.detectAndCompute(gray,None)
    contours = []
    detected_card_ids = []
    for card_id in card_features.keys():
        ret, contour = detect_card_in_image(image, kp, des, card_id)
        
        if ret > 0:
            #card found
            contours.append(contour)
            detected_card_ids.append(card_id)
            print "card found " + card_id

    if len(detected_card_ids) > 0:
        #check the winning card
        winning_card_id = compare_cards(detected_card_ids)
        print winning_card_id + " wins"

        for i, c in enumerate(contours):

            #determine the card contour's center http://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html#gsc.tab=0
            M = cv2.moments(c)

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            if winning_card_id == detected_card_ids[i]:
                cx = cx - (winnerTextSize[0] / 2)
                cy = cy - (winnerTextSize[1] / 2)
                cv2.putText(image, "WINNER", (cx , cy), fontFace, fontScale, (0, 255, 0), thickness, 8);
                cv2.polylines(image,[np.int32(c)],True,(0,255,0),3, cv2.LINE_AA)
            else:
                cx = cx - (loserTextSize[0] / 2)
                cy = cy - (loserTextSize[1] / 2)
                cv2.putText(image, "LOSER", (cx, cy), fontFace, fontScale, (0, 0, 255), thickness, 8);
                cv2.polylines(image,[np.int32(c)],True,(0,0,255),3, cv2.LINE_AA)

    cv2.imshow("final", image)
    cv2.waitKey(0)

#load the database
load_cards()

#read image from argument
image = cv2.imread(sys.argv[1])

detect_cards(image)



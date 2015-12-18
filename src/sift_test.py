import cv2
import cv2.xfeatures2d
import numpy as np
import sys
import store_keypoints as skp
import cPickle as pickle

card_values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "queen", "jack", "king", "ace"]
card_suits = ["clubs", "diamonds", "hearts", "spades"]
card_images = dict()
card_features = dict()
MIN_MATCH_COUNT = 100
# Initiate SIFT detector
SIFT_DETECTOR = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10, 1.6)


'''def compare_cards(card_ids):
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

    return winning_id'''


def extract_features_sift(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return SIFT_DETECTOR.detectAndCompute(gray, None)


def load_cards():
    for suit in card_suits:
        for value in card_values:
            card_id = value + "_" + suit
            card_images[card_id] = cv2.imread("database/" + card_id + ".png")
            card_features[card_id] = skp.unpickle_keypoints(pickle.load(open("database/" + card_id + ".sift")))


def start_camera_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', card_search(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_card_in_image(acquired_image, acquired_image_kp, acquired_image_des, card_id):
    img1 = card_images[card_id]
    img2 = cv2.cvtColor(acquired_image, cv2.COLOR_BGR2GRAY)   

    # find the keypoints and descriptors with SIFT
    kp1, des1 = card_features[card_id]
    kp2 = acquired_image_kp
    des2 = acquired_image_des
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        
        #image_with_square = cv2.polylines(acquired_image,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        return None
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                      matchesMask = matchesMask, # draw only inliers
                     flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv2.imshow("result", img3)
    return dst


def card_search(image):
    kp, des = SIFT_DETECTOR.detectAndCompute(image,None)
    cardsWithContour = dict()
    for suit in card_suits:
        for value in card_values:
            card_id = value + "_of_" + suit 
            contour_points = detect_card_in_image(image, kp, des, card_id)
            if(not contour_points is None):
                print("Found " + card_id)
                cardsWithContour[card_id] = contour_points
                #finalImage = cv2.polylines(finalImage,[np.int32(contour_points)],True,255,3, cv2.LINE_AA)
    return cardsWithContour

def find_and_draw(image):
    for suit in card_suits:
        for value in card_values:
            card_id = value + "_of_" + suit 
            contour_points = detect_card_in_image(image, card_id)
            if(contour_points != None):
                print("Found " + card_id)
                img_copy = image
                cv2.imshow(card_id, cv2.polylines(img_copy,[np.int32(contour_points)],True,255,3, cv2.LINE_AA))


load_cards()

image = cv2.imread(sys.argv[1])
cardsWithContour = card_search(image)
winning_card_id = compare_cards(cardsWithContour.keys())
contour_points = cardsWithContour[winning_card_id]
cv2.imshow("final", cv2.polylines(image,[np.int32(contour_points)],True,255,3, cv2.LINE_AA))

#start_camera_feed()
cv2.waitKey(0)


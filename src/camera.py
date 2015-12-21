import numpy as np
import cv2
import time
import card_detector as cd

def check_image_difference(img1, img2):

    diff = cv2.absdiff(last_frame, frame)
    cv2.imshow("diff", diff)
    return


cap = cv2.VideoCapture(0)
cd.load_cards()

#ret, last_frame = cap.read()
#ret, one_frame = cap.read()
#last_time = time.time()
#temp_diff = cv2.absdiff(last_frame, one_frame)
#last_diff = temp_diff.sum()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)

#    if (time.time()-last_time) > 5:
#        diff = cv2.absdiff(last_frame, frame)
#        diff_sum = diff.sum()

#        print min(last_diff,diff_sum)/float(max(last_diff, diff_sum))
#        
#        if min(last_diff,diff_sum)/float(max(last_diff, diff_sum)) > 0.50:
#            cv2.imshow("diff", diff)
#            last_diff = diff_sum
#
#        last_time = time.time()
#        last_frame = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("detecting...")
        image = frame
        cd.detect_cards(image)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
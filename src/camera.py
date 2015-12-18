import numpy as np
import cv2

def check_image_difference(img1, img2):
	diff_img = cv2.absdiff(img1, img2);

    cv::Mat foregroundMask = cv::Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);

    float threshold = 30.0f;
    float dist;

    for(int j=0; j<diffImage.rows; ++j)
        for(int i=0; i<diffImage.cols; ++i)
        {
            cv::Vec3b pix = diffImage.at<cv::Vec3b>(j,i);

            dist = (pix[0]*pix[0] + pix[1]*pix[1] + pix[2]*pix[2]);
            dist = sqrt(dist);

            if(dist>threshold)
            {
                foregroundMask.at<unsigned char>(j,i) = 255;
            }
        }

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    cv2.imshow('threshold',threshold)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
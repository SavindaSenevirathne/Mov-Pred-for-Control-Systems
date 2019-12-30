import numpy as np
import cv2
from Pose_Detection import PoseDetection

frame = cv2.imread('media/sitting2.JPG')
if frame is not None:
    height, width, layers = frame.shape
    if((height > 700) | (width > 1366)):
        frame = cv2.resize(frame, (int(width/2), int(height/2)), interpolation=cv2.INTER_AREA)
        
frame1, frame2 = PoseDetection.detectPose(frame)

cv2.imshow('Ske', frame1)
cv2.imshow('Points', frame2)
cv2.imwrite('output-frame1.jpg', frame1)
cv2.imwrite('output-frame2.jpg', frame2)

cv2.waitKey(0)


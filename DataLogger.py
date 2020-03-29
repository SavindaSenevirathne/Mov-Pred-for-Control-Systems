import cv2
import numpy as np
from Pose_Detection import PoseDetection

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture("media/common2.MOV")

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    if frame is not None:
        height, width, layers = frame.shape
        if((height > 700) | (width > 1366)):
            frame = cv2.resize(frame, (int(width/2), int(height/2)), interpolation=cv2.INTER_AREA)

    frame1, frame2 = PoseDetection.detectPose(frame, True)
    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

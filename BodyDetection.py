import numpy as np 
import cv2

minSize_w = 60 # 60
minSize_h = 100 # 130

maxSize_w = 200 #120
maxSize_h = 300 # 250
# lady walking
# height = 130
# width = 60

def detect(frame):
    global points
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # body = fullBody_cascade.detectMultiScale(
    #     frame_gray, minSize=(minSize_w, minSize_h), maxSize=(maxSize_w, maxSize_h))
    body = upperBody_cascade.detectMultiScale(
        frame_gray, minSize=(minSize_w, int(minSize_h/2)), maxSize=(maxSize_w, int(maxSize_h/2)))
    # lower_body = lowerBody_cascade.detectMultiScale(
    #     frame_gray, minSize=(minSize_w, int(minSize_h/2)), maxSize=(maxSize_w, int(maxSize_h/2)))
    # print(body, upper_body, lower_body)


    for (x, y, w, h) in body:
        # print(np.append(points, np.array([[1,1]])))
        points = np.append(points, np.array([[x+int(w/2), y]]), axis=0)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    # for (x, y, w, h) in upper_body:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # for (x, y, w, h) in lower_body:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    if (points.size > 0):
        cv2.polylines(frame, [points], False, (255,0,0), 2)
    return frame


fullBody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_fullbody.xml')
upperBody_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_upperbody.xml')
lowerBody_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + '/haarcascade_lowerbody.xml')

# cap = cv2.VideoCapture(0)
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# cap = cv2.VideoCapture('./media/lady_walking.mp4')
# cap = cv2.VideoCapture('./media/lobby.mp4')
cap = cv2.VideoCapture('./media/common2.MOV')
# cap = cv2.VideoCapture('./media/two_men.mp4')
# cap = cv2.VideoCapture('./media/cricket_2.mp4')
global points
points = np.empty((0, 2), int)
while 1:
    # global points
    ret, frame = cap.read()
    if frame is not None:
        height, width, layers = frame.shape
        if((height > 700) | (width > 1366)):
            frame = cv2.resize(frame, (int(width/2), int(height/2)),
                            interpolation=cv2.INTER_AREA)
    # fgmask = fgbg.apply(frame)
    frame = detect(frame)
    cv2.imshow('detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'): # wait 40 ms per frame to match 25 fps of the video
        break

cap.release()
cv2.destroyAllWindows()

# frame = cv2.imread('./media/images/IMG_5425.JPG')
# frame = detect(frame)
# frame = cv2.resize(frame, (500, 800), interpolation=cv2.INTER_AREA)

# cv2.imshow('test', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



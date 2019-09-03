import numpy as np 
import cv2

def detect(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # fgmask = fgbg.apply(frame)

    # body = fullBody_cascade.detectMultiScale(frame_gray)
    body = upperBody_cascade.detectMultiScale(frame_gray)
    # body = lowerBody_cascade.detectMultiScale(frame_gray)
    print(body)
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 5)
    return frame


fullBody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_fullbody.xml')
upperBody_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_upperbody.xml')
lowerBody_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + '/haarcascade_lowerbody.xml')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

cap = cv2.VideoCapture(0)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
cap = cv2.VideoCapture('./media/lady_walking.mp4')
# cap = cv2.VideoCapture('./media/lobby.mp4')
# cap = cv2.VideoCapture('./media/man_walking.mp4')
# cap = cv2.VideoCapture('./media/two_men.mp4')

while 1:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1366, 700), interpolation=cv2.INTER_AREA)

    # fgmask = fgbg.apply(frame)
    frame = detect(frame)
    cv2.imshow('detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# frame = cv2.imread('./media/images/IMG_5425.JPG')
# frame = detect(frame)
# frame = cv2.resize(frame, (500, 800), interpolation=cv2.INTER_AREA)

# cv2.imshow('test', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



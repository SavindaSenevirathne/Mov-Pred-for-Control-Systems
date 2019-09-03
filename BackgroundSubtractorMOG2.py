import numpy as np
import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./media/IMG_5403_Trim.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()

i = 1

while(1):
    ret, frame = cap.read()
    # frame = cv2.flip(frame, -1)
    frame = cv2.resize(frame, (1366, 700), interpolation=cv2.INTER_AREA)
    fgmask = fgbg.apply(frame)
    cv2.imshow('BackgroundSubtractorMOG2', fgmask)

    # if cv2.waitKey(1) & 0xFF == ord('c'):
    #     print('Capturing', i)
    #     cv2.imwrite('BackgroundSubtractorMOG2' + str(i) + '.jpg', fgmask)
    #     i = i + 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

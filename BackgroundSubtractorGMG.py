import numpy as np
import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./media/IMG_5403_Trim.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
i = 1
while(1):
    ret, frame = cap.read()
    # frame = cv2.flip(frame, -1)
    frame = cv2.resize(frame, (1366, 700), interpolation=cv2.INTER_AREA)
    # frame = cv2.flip(frame, -1)
    # frame = cv2.rotate(frame, 1)
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('BackgroundSubtractorGMG', fgmask)
    
    # if cv2.waitKey(1) & 0xFF == ord('c'):
    #     print('Capturing', i)
    #     cv2.imwrite('BackgroundSubtractorGMG' + str(i) + '.jpg', fgmask)
    #     i = i + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

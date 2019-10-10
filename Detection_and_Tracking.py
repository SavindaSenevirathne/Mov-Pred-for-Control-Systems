import numpy as np
import cv2

# Variables
minSize_w = 60  # 60
minSize_h = 100  # 130

maxSize_w = 200  # 120
maxSize_h = 300  # 250


# Detect human
def detect(frame):
    objectDetected = False
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    body = fullBody_cascade.detectMultiScale(
        frame_gray, minSize=(minSize_w, minSize_h), maxSize=(maxSize_w, maxSize_h))
    detectedObjects = []
    for (x, y, w, h) in body:
        objectDetected = True
        frameC = frame
        cv2.rectangle(frameC, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frameC, 'Detected Object', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Detection', frameC)
        cv2.imwrite('detected-object.jpg', frameC)
        detectedObjects.append((x,y,(x+w),(y+h)))
    return frame, objectDetected, detectedObjects

def initializeTracking():
    print('Start tracking..')
    for box in detectedObjects:
        # tracker = cv2.TrackerCSRT_create()
        tracker = cv2.TrackerMOSSE_create()

        MultiTracker.add(tracker, frame, box)

    return True

    

# Main method
if __name__ == "__main__":
    fullBody_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + '/haarcascade_fullbody.xml')
    # fullBody_cascade = cv2.CascadeClassifier(
    #     cv2.data.haarcascades + '/haarcascade_frontalface_alt.xml')

    # videoSouce = 0
    videoSouce = 'media/cricket_2.mp4'
    cap = cv2.VideoCapture(videoSouce)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    isObjectDetected = False
    detectedObjects = []
    MultiTracker = cv2.MultiTracker_create()
    isTrackerAdded = False

    while True:
        ret, frame = cap.read()
        # resize if the frame size is large
        height, width, layers = frame.shape
        if((height > 700) | (width > 1366)):
          frame = cv2.resize(frame, (int(width/2), int(height/2)), interpolation=cv2.INTER_AREA)

        if not isObjectDetected:
            frameC, isObjectDetected, detectedObjects = detect(frame)
            print('detected objects from the source', detectedObjects)

        if not isTrackerAdded and isObjectDetected:
            # start tracking
            isTrackerAdded = initializeTracking()

        success, boxes = MultiTracker.update(frame)

        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]/2), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.putText(frame, 'Tracking',p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        if(success):
            cv2.imshow('Tracking', frame)

        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

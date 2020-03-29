import numpy as np
import cv2
from Pose_Detection import PoseDetection
from opencvFaceRecognition.recognize_person import detectPerson
# from Pose_Detection_Multi import PoseDetection

def waitAndClose(list):
	while True:
		waitKey = cv2.waitKey(10)
		if waitKey == ord('x'):
			for win in list:
				cv2.destroyWindow(win)
			break

def detectSkeleton(frame):
	# obj = PoseDetection()
	frame1, frame2 = PoseDetection.detectPose(frame, False)
	# cv2.imwrite('output-frame1.jpg', frame1)
	# cv2.imwrite('output-frame2.jpg', frame2)
	cv2.imshow('Pose skeleton', frame1)
	# cv2.imshow('Pose skeleton points', frame2)
	waitAndClose(['Pose skeleton', 'Pose skeleton points'])


# Detect human
def detect(personName, frame):
	isObjectDetected = False
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_gray = cv2.equalizeHist(frame_gray)
	body = cascade.detectMultiScale(
		frame_gray, minSize=(minSize_w, minSize_h), maxSize=(maxSize_w, maxSize_h))
	detectedObjects = []
	for (x, y, w, h) in body:
		isObjectDetected = True
		frameC = frame
		cv2.rectangle(frameC, (x, y), (x+w, y+h), (255, 0, 0), 2)
		cv2.putText(frameC, 'Detected Person: ' + personName, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
					1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
		cv2.imshow('Detection', frameC)
		# cv2.imwrite('detected-object.jpg', frameC)
		detectedObjects.append((x,y,(x+w),(y+h)))
		waitAndClose(['Detection'])

	return frame, isObjectDetected, detectedObjects

def initializeTracking():
	print('Start tracking..')
	for box in detectedObjects:
		# tracker = cv2.TrackerCSRT_create()
		tracker = cv2.TrackerMOSSE_create()

		MultiTracker.add(tracker, frame, box)

	return True

	

# Main method
if __name__ == "__main__":
	# Variables
	minSize_w = 60  # 60
	minSize_h = 100  # 100

	maxSize_w = 200  # 120
	maxSize_h = 300  # 250
	cascade = cv2.CascadeClassifier(
		cv2.data.haarcascades + '/haarcascade_upperbody.xml')
	# cascade = cv2.CascadeClassifier(
	#     cv2.data.haarcascades + '/haarcascade_frontalface_alt.xml')

	# face detection section
	faceCapture = cv2.VideoCapture(cv2.CAP_DSHOW)
	if not faceCapture.isOpened:
		print('--(!)Error opening video capture')
		exit(0) 
	print('Press X when satisfied with the recognition')
	while True:
		ret, faceFrame = faceCapture.read()
		# resize if the faceFrame size is large
		if faceFrame is not None:
			height, width, layers = faceFrame.shape
			if((height > 700) | (width > 1366)):
				faceFrame = cv2.resize(faceFrame, (int(width/2), int(height/2)), interpolation=cv2.INTER_AREA)
			
		faceFrame, recognizedPerson = detectPerson().detect_face(faceFrame)
		cv2.imshow('Tracking', faceFrame)

		if faceFrame is None:
			print('--(!) No captured frame -- Break!')
			break

		waitKey = cv2.waitKey(10)
		if waitKey == ord('x'):
			# cv2.destroyWindow('Tracking')
			break

	# videoSouce = 0
	videoSouce = 'media/common2.MOV'

	# user inputs
	# print('Common room     [0]')
	# print('Aruna Commmon 1 [1]')
	# print('Aruna Commmon 2 [2]')
	# print('Live Stream     [3]')
	# userInput = input('Choose the video source: ')
	# x = int(userInput)
	# if  x == 0:
	# 	print('Common room') # default option
	# elif x == 1:
	# 	print('Aruna Commmon 1')
	# 	minSize_w = 30  # 60
	# 	minSize_h = 50  # 100
	# 	videoSouce = 'media/aruna1.MOV'
	# elif x == 2:
	# 	print('Aruna Commmon 2')
	# 	videoSouce = 'media/aruna2.MOV'
	# elif x == 3:
	# 	print('Live stream')
	# 	videoSouce = 'rtsp://admin:abcd@1234@192.168.8.101/'

	cap = cv2.VideoCapture(videoSouce)
	if not cap.isOpened:
		print('--(!)Error opening video capture')
		exit(0)

	isObjectDetected = False
	detectedObjects = []
	MultiTracker = cv2.MultiTracker_create()
	isTrackerAdded = False
	print('Detecting..')
	while True:
		ret, frame = cap.read()
		# resize if the frame size is large
		if frame is not None:
			height, width, layers = frame.shape
			if((height > 700) | (width > 1366)):
				frame = cv2.resize(frame, (int(width/2), int(height/2)), interpolation=cv2.INTER_AREA)

		if not isObjectDetected:
			frameC, isObjectDetected, detectedObjects = detect(recognizedPerson, frame)
			if len(detectedObjects) > 0:
				print('detected objects from the source', detectedObjects)
		if not isTrackerAdded and isObjectDetected:
			# start tracking
			isTrackerAdded = initializeTracking()

		cleanFrame = np.copy(frame)
		
		success, boxes = MultiTracker.update(frame)

		for i, newbox in enumerate(boxes):
			p1 = (int(newbox[0]), int(newbox[1]))
			p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
			# p2 = (int(newbox[2]), int(newbox[3]))
			cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
			cv2.putText(frame, 'Tracking: ' + recognizedPerson,p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

		if(success):
			cv2.imshow('Tracking', frame)

		if frame is None:
			print('--(!) No captured frame -- Break!')
			break

		waitKey = cv2.waitKey(10)
		if waitKey == ord('q'):
			break
		elif waitKey == ord('c'):
			detectSkeleton(cleanFrame)

	faceCapture.release()
	cap.release()
	cv2.destroyAllWindows()

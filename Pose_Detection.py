import cv2
import time
import numpy as np

""" 
Coco dataset data points
0 	- 	Nose
1 	- 	Neck
2 	- 	Right Shoulder
3 	- 	Right Elbow
4 	- 	Right Wrist
5 	-	Left Shoulder
6 	- 	Left Elbow
7 	- 	Left Wrist
8 	- 	Right Hip
9 	- 	Right Knee
10 	- 	Right Ankle
11 	- 	Left Hip
12 	- 	Left Knee
13 	- 	LAnkle
14 	- 	Right Eye
15 	- 	Left Eye
16 	- 	Right Ear 
17 	- 	Left Ear
18 	- 	Background

 """

class PoseDetection:
	"Pose Detection happens here"

	@staticmethod
	def detectPose(frame):

		protoFile = "pose/coco/pose_deploy_linevec.prototxt"
		weightsFile = "pose/coco/pose_iter_440000.caffemodel"
		nPoints = 18
		POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [
			8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

		frameCopy = np.copy(frame)
		frameWidth = frame.shape[1]
		frameHeight = frame.shape[0]
		threshold = 0.1

		net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

		t = time.time()
		# input image dimensions for the network
		inWidth = 368
		inHeight = 368
		inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
										(0, 0, 0), swapRB=False, crop=False)

		net.setInput(inpBlob)

		output = net.forward()
		print("time taken by network : {:.3f}".format(time.time() - t))
		H = output.shape[2]
		W = output.shape[3]

		# Empty list to store the detected keypoints
		points = []

		for i in range(nPoints):
			# confidence map of corresponding body's part.
			probMap = output[0, i, :, :]

			# Find global maxima of the probMap.
			minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

			# Scale the point to fit on the original image
			x = (frameWidth * point[0]) / W
			y = (frameHeight * point[1]) / H

			if prob > threshold:
				cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255),
						thickness=-1, lineType=cv2.FILLED)
				cv2.putText(frameCopy, "{}".format(i), (int(x), int(
					y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

				# Add the point to the list if the probability is greater than the threshold
				points.append((int(x), int(y)))
			else:
				points.append(None)
		print('points', points)

		if points[0] and points[8] and points[10] and points[11] and points[9] and points[12] and points[13] is not None :
			# print('Nose - ', points[0])
			# print('R hip - ', points[8])
			# print('L hip - ', points[11])
			# print('R Knee - ', points[9])
			# print('L Knee - ', points[12])
			# print('R anckle - ', points[10])			
			# print('L anckle - ', points[13])


			_, nose = points[0]
			_, rHip = points[8]
			_, lHip = points[11]
			_, rKnee = points[9]
			_, lKnee = points[12]
			_, rAncle = points[10]			
			_, lAncle = points[13]

			hip = (rHip+lHip)/2
			knee = (rKnee+lKnee)/2
			ancle = (rAncle+lAncle)/2
			upperToLower = (hip - nose)/(ancle - hip)
			kneeToUpper = (knee - hip)/(hip - nose)
			print('Nose to ancle: ', ancle - nose)
			print('Nose to hip: ', hip - nose)
			print('Hip to knee: ', knee - hip)
			print('Hip to ancle: ', ancle - hip)
			print('upperbody/lowerbody: ', upperToLower)
			print('hipToKnee/upperbody: ', kneeToUpper)

			if kneeToUpper >= 0.5:
				cv2.putText(frame, 'Standing', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 
							1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
				cv2.putText(frameCopy, 'Standing', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 
							1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
				print('Person is standing')
			else:
				cv2.putText(frame, 'Sitting', (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
				            1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
				cv2.putText(frameCopy, 'Sitting', (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
				            1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
				print('Person is sitting')
		else:
			print('No enough points to identify the posture')

		# Draw Skeleton
		for pair in POSE_PAIRS:
			partA = pair[0]
			partB = pair[1]

			if points[partA] and points[partB]:
				cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
				cv2.circle(frame, points[partA], 8, (0, 0, 255),
						thickness=-1, lineType=cv2.FILLED)

		print("Total time taken : {:.3f}".format(time.time() - t))

		return frame, frameCopy

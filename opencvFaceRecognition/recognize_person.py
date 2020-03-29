from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

class detectPerson:
	"face detection happens here"
	print("[INFO] loading face detector...")
	protoPath = os.path.sep.join(["opencvFaceRecognition", "face_detection_model", "deploy.prototxt"])
	modelPath = os.path.sep.join(["opencvFaceRecognition", "face_detection_model",
		"res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# load our serialized face embedding model from disk
	print("[INFO] loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch("opencvFaceRecognition/openface_nn4.small2.v1.t7")

	# load the actual face recognition model along with the label encoder
	recognizer = pickle.loads(open("opencvFaceRecognition/output/recognizer.pickle", "rb").read())
	le = pickle.loads(open("opencvFaceRecognition/output/le.pickle", "rb").read())

	def detect_face(self, givenFrame):
		personName = ""
	
		frame = imutils.resize(givenFrame, width=600)
		(h, w) = frame.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		self.detector.setInput(imageBlob)
		detections = self.detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				self.embedder.setInput(faceBlob)
				vec = self.embedder.forward()

				# perform classification to recognize the face
				preds = self.recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = self.le.classes_[j]
				# if name is not None:
				# 	personName = name

				# draw the bounding box of the face along with the
				# associated probability
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		return frame, personName

# USAGE
# python photo_booth.py --output output

# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
import argparse
import time
import cv2

from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import os
from Pose_Detection import PoseDetection
from opencvFaceRecognition.recognize_person import detectPerson
from opencvFaceRecognition.extact_embeddings_person import extractEmbeddings
from opencvFaceRecognition.train_model_person import trainModel



class Interface:
	def __init__(self, vs, outputPath):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.vs = vs
		self.outputPath = outputPath
		self.frame = None
		self.thread = None
		self.stopEvent = None

		# model training thread
		self.modelTrainingThread = None

		# initialize the root window and image panel
		self.root = tki.Tk()
		self.panel = None

		# create a button, that when pressed, will take the current
		# frame and save it to file
		self.btn1 = tki.Button(self.root, text="Capture",
							   command=self.takeSnapshot, state="disabled", font=("Times", 15, "bold"))
		self.btn1.grid(row=2, column=4, padx=5, pady=5)

		self.btn2 = tki.Button(
			self.root, text="Start", command=self.startSkeletonDetection, state="disabled", font=("Times", 15, "bold"))
		self.btn2.grid(row=2, column=5, padx=5, pady=5)

		self.btn3 = tki.Button(self.root, text="Confirm",
							   command=self.confirmRecognition, font=("Times", 15, "bold"))
		self.btn3.grid(row=2, column=6, padx=5, pady=5)

		self.personNameVar = tki.StringVar()
		self.personNameVar.set('Recognized Person: ')
		nameLabel = tki.Label(self.root, textvariable=self.personNameVar, font=("Times", 15, "bold"))
		nameLabel.grid(row=1, column=1, columnspan=9, padx=5, pady=5)

		self.instruction = tki.StringVar()
		self.instruction.set('Please click confirm button to confirm...')
		nameLabel = tki.Label(self.root, textvariable=self.instruction,fg="blue", font=("Times", 14, "bold"))
		nameLabel.grid(row=3, column=1, columnspan=9, padx=5, pady=5)


		# start a thread that constantly pools the video sensor for
		# the most recently read frame
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

		# set a callback to handle when the window is closed
		self.root.wm_title("Movement Prediction")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
		self.performFaceRecognition = True
		self.performSkeletonDetection = False

	def videoLoop(self):
		# DISCLAIMER:
		# I'm not a GUI developer, nor do I even pretend to be. This
		# try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading
		currentNumber = 0
		try:
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				# grab the frame from the video stream and resize it to
				# have a maximum width of 300 pixels
				_, self.frame = self.vs.read()
				# self.frame = imutils.resize(self.frame, width=600)
				# self.frame = cv2.resize(
				#     self.frame, int(600), interpolation=cv2.INTER_AREA)
				if self.frame is None:
					print('--(!) No captured frame -- Break!')
					break

				h, w, _ = self.frame.shape
				if((h > 700) | (w > 1366)):
					self.frame = cv2.resize(
						self.frame, (int(w/2), int(h/2)), interpolation=cv2.INTER_AREA)
				# perform face recognition on button press
				if (self.performFaceRecognition):
					self.frame, self.detectedPersonName = detectPerson().detect_face(self.frame)
					self.personNameVar.set(
						'Recognized Person: ' + self.detectedPersonName)
				# perform skeleton detection on button press
				if (self.performSkeletonDetection):
					if currentNumber % 30 == 0:
						self.frame, frameCopy = PoseDetection.detectPose(
							self.frame, False)
					currentNumber += 1

				# OpenCV represents images in BGR order; however PIL
				# represents images in RGB order, so we need to swap
				# the channels, then convert to PIL and ImageTk format
				image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(image)
				image = ImageTk.PhotoImage(image)
  
				# if the panel is not None, we need to initialize it
				if self.panel is None:
					self.panel = tki.Label(image=image)
					self.panel.image = image
					self.panel.grid(row=0, column=0,
									columnspan=11, padx=5, pady=5)

				# otherwise, simply update the panel
				else:
					self.panel.configure(image=image)
					self.panel.image = image

		except RuntimeError:
			print("[INFO] caught a RuntimeError")

	def takeSnapshot(self):
		ret, frame = self.vs.read()
		img_name = "{}.png".format(self.img_counter)
		cv2.imwrite(os.path.join(self.path, img_name), frame)
		self.img_counter += 1
		if(self.img_counter > 5):
			self.btn2["state"] = "active"
			self.instruction.set('You have captured '+str(self.img_counter)+'(>5) images. If you want, you can start now...')
		else:
			self.instruction.set('Please capture more than 5 images. You have taken only '+str(self.img_counter)+' images...')

	def startSkeletonDetection(self):
		self.btn1["state"] = "disabled"
		self.btn2["state"] = "disabled"
		if(self.detectedPersonName == "unknown"):
			self.personNameVar.set('Newly added Person: ' +self.directory)
			self.instruction.set('Training the face data set with newly added data....')
			self.modelTrainingThread = threading.Thread(target=self.trainFacialModel, args=())
			self.modelTrainingThread.start()
		self.instruction.set('')
		self.vs.release()
		self.vs = cv2.VideoCapture('media/common2.MOV')
		self.performFaceRecognition = False
		self.performSkeletonDetection = True

	def confirmRecognition(self):
		self.performFaceRecognition = False
		self.frame, self.detectedPersonName = detectPerson().detect_face(self.frame)
		self.personNameVar.set(
		'Recognized Person: ' + self.detectedPersonName)
		# check detected person is unkonwn
		if(self.detectedPersonName == "unknown"):
			self.img_counte = 0
			self.btn3["state"] = "disabled"
			self.btn1["state"] = "active"
			todayTime = datetime.datetime.now().strftime("%Y%b%d_%H%M%S")
			self.img_counter = 0
			parent_dir = "opencvFaceRecognition/dataset/"
			self.directory = todayTime
			self.path = os.path.join(parent_dir, self.directory)
			os.mkdir(self.path)
			self.instruction.set('Please capture more than 5 images. You have taken only '+str(self.img_counter)+' images...')
		# check detected person is null
		elif(self.detectedPersonName == ""):
			self.performFaceRecognition = True
			self.instruction.set('Please click confirm button to confirm after detecting as a person...')
		else:
			self.btn3["state"] = "disabled"
			self.btn2["state"] = "active"
			self.instruction.set('Recognize the person as '+self.detectedPersonName+'. You can start now...')

	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		print("[INFO] releasing camera..")
		self.vs.release()
		print("[INFO] closing window..")
		self.root.destroy()
		if self.modelTrainingThread is not None:
			print("[INFO] stoping training thread..")
			self.modelTrainingThread._stop()
		exit(0)

	def trainFacialModel(self):
		extractEmbeddings().embedding()
		trainModel().training()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="output",
				help="path to output directory to store snapshots")
ap.add_argument("-p", "--picamera", type=int, default=0,
				help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
# vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
vs = cv2.VideoCapture(cv2.CAP_DSHOW)
time.sleep(2.0)

# start the app
pba = Interface(vs, args["output"])
pba.root.mainloop()
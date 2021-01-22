import numpy as np
import time
import cv2
import imutils
import sys
from imutils.video import FPS
from imutils.video import VideoStream

INPUT_FILE='/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/data/test_video_short.avi'
OUTPUT_FILE='/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/data/result/output.avi'
LABELS_FILE='/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/data/obj.names'
CONFIG_FILE='/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/cfg/yolov3.cfg'
WEIGHTS_FILE='/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/backup/yolov3_last.weights'
CONFIDENCE_THRESHOLD=0.3

H=None
W=None
img_b_2f=None
img_b_1f=None
img_accident=None
img_a_1f=None
img_a_2f=None
img_a_10f=None
img_org=None
image=None

#상수 값 Class ID
FACE = 0
ACCIDENT = 1
DRIVE = 2

fps = FPS().start()

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30,
	(1280, 720), True)

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
vs = cv2.VideoCapture(INPUT_FILE)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
cnt =0;
while True:
	cnt+=1
	#print ("Frame number", cnt)
	try:
		# Step0 항상 이전 2프레임과 현재 프레임은 저장해놓음
		#	img_b_2f = img_b_1f # 2프레임 이전 이미지
		#	img_b_1f = img_org # 1프레임 이전 이미지
		(grabbed, image) = vs.read() # 현재 프레임 이미지
	except:
		break

	img_org = image

	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	if W is None or H is None:
		(H, W) = image.shape[:2]
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE_THRESHOLD:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

				# Step1 사고 발생 감지 후 이미지 저장
				if classID == ACCIDENT:

					img_accident = image

					fps.update()
					(grabbed, image) = vs.read()
					img_a_1f = image

					fps.update()
					(grabbed, image) = vs.read()
					img_a_2f = image

					fps.update()
					fps.update()
					fps.update()
					fps.update()
					fps.update()
					fps.update()
					fps.update()
					fps.update()
					(grabbed, image) = vs.read()
					img_a_10f = image


					cv2.imwrite('/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/data/result/img_b_2f.jpg', img_b_2f)
					cv2.imwrite('/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/data/result/img_b_1f.jpg', img_b_1f)
					cv2.imwrite('/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/data/result/img_accident.jpg', img_accident)
					cv2.imwrite('/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/data/result/img_a_1f.jpg', img_a_1f)
					cv2.imwrite('/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/data/result/img_a_2f.jpg', img_a_2f)
					cv2.imwrite('/Users/kimjongseok/PycharmProjects/design_Rev1/darknet/data/result/img_a_10f.jpg', img_a_10f)
					print("Done saving accidnet images!")
					sys.exit(1)


	# face 클래스 컨테이너 박스 개수 구하기
	count_face_image = 0
	for index in classIDs:
		if index == 0:
			count_face_image += 1

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
		CONFIDENCE_THRESHOLD)


	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[classIDs[i]]]

			# 컨테이너 박스 그리기
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			# 컨테이너 박스 위에 클래스명과 정확도 표기
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.imshow('test', cv2.resize(image,(1280, 720)))

			#if classIDs[i]
			cv2.putText(image, text, (x+100, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# show the output image
	#cv2.imshow("output", cv2.resize(image,(1280, 720)))
	#writer.write(cv2.resize(image,(1280, 720)))

	img_b_2f = img_b_1f # 2프레임 이전 이미지
	img_b_1f = img_org # 1프레임 이전 이미지

	fps.update()
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
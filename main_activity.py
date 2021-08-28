import numpy as np
import time
import cv2
import imutils
import sys
import accident_activity
import head_pose_activity
from imutils.video import FPS
from imutils.video import VideoStream

INPUT_FILE = '/Users/kimjongseok/PycharmProjects/Accident_Identification_RevA/testgogo.mov'
OUTPUT_FILE = '/Users/kimjongseok/PycharmProjects/Accident_Identification_RevA/result/video/output.avi'
LABELS_FILE = '/Users/kimjongseok/PycharmProjects/Accident_Identification_RevA/darknet/data/obj.names'
CONFIG_FILE = '/Users/kimjongseok/PycharmProjects/Accident_Identification_RevA/darknet/cfg/yolov3-tiny.cfg'
WEIGHTS_FILE = '/Users/kimjongseok/PycharmProjects/Accident_Identification_RevA/darknet/backup/yolov3_last.weights'
CONFIDENCE_THRESHOLD=0.3

OUTPUTYOLO = '/Users/kimjongseok/PycharmProjects/Accident_Identification_RevA/result/others/yolo_accident_value.txt'
OUTPUTHEAD = '/Users/kimjongseok/PycharmProjects/Accident_Identification_RevA/result/others/head_pose_estimation_value.txt'

accident_activity.IdentifyActivity(INPUT_FILE, OUTPUT_FILE, LABELS_FILE, CONFIG_FILE, WEIGHTS_FILE, OUTPUTYOLO, CONFIDENCE_THRESHOLD)
head_pose_activity.HeadPoseEstimation(INPUT_FILE, OUTPUT_FILE, OUTPUTHEAD)

print("done")
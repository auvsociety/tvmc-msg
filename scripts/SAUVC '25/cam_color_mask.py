#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
from tvmc import MotionController, DoF, ControlMode
import time




def create_pipeline():
	pipeline = dai.Pipeline()
	cam = pipeline.create(dai.node.ColorCamera)
	xout = pipeline.create(dai.node.XLinkOut)
	xout.setStreamName("video")
	
	cam.setPreviewSize(640,640)
	cam.setPreviewKeepAspectRatio(False)
	cam.setInterleaved(False)
	cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
	cam.setFps(30)
	cam.preview.link(xout.input)
	return pipeline

def detect_flares(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_red = np.array([0,90,90])
	upper_red = np.array([15,255,255])
	lower_yellow = np.array([15,100,100])
	upper_yellow = np.array([50,255,255])
	lower_blue = np.array([90,100,100])
	upper_blue = np.array([130,255,255])
	
	masks = {
		"red": cv2.inRange(hsv, lower_red, upper_red),
		#"blue": cv2.inRange(hsv, lower_blue, upper_blue),
		"yellow": cv2.inRange(hsv, lower_yellow, upper_yellow)
	}
	
	detections = []
	
	for color, mask in masks.items():
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > 500:
				x,y,w,h = cv2.boundingRect(cnt)
				cx,cy = x+w //2, y+h//2
				detections.append((cx,cy,color))
				cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
				cv2.putText(frame,color,(x,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0,2))
				
	return frame, detections


def main():
	with dai.Device(create_pipeline()) as device:
		qvideo = device.getOutputQueue(name = "video", maxSize = 4, blocking = False)
		
		while True:
			inVideo = qvideo.get()
			frame = inVideo.getCvFrame()
			p_frame,detections = detect_flares(frame)
			
			cv2.imshow("flares", p_frame)
			if cv2.waitKey(1) == ord('q'):
				break
			


if __name__ == "__main__":

	main()
	
	
	
	
	
	
	
	
	
	
	
	

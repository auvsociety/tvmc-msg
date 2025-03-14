#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np

def nothing(x):
    pass

def create_trackbars():
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("Blur", "Trackbars", 1, 20, nothing)
    cv2.createTrackbar("Erode", "Trackbars", 0, 5, nothing)
    cv2.createTrackbar("Dilate", "Trackbars", 0, 5, nothing)

def get_trackbar_values():
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")
    blur = cv2.getTrackbarPos("Blur", "Trackbars")
    erode = cv2.getTrackbarPos("Erode", "Trackbars")
    dilate = cv2.getTrackbarPos("Dilate", "Trackbars")
    
    blur = max(1, blur) if blur % 2 == 1 else max(1, blur + 1)
    return (h_min, h_max, s_min, s_max, v_min, v_max, blur, erode, dilate)

def process_frame(frame):
    h_min, h_max, s_min, s_max, v_min, v_max, blur, erode, dilate = get_trackbar_values()
    
    frame = cv2.GaussianBlur(frame, (blur, blur), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=erode)
    mask = cv2.dilate(mask, kernel, iterations=dilate)
    
    return mask, lower_bound, upper_bound, hsv

def create_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    
    cam.setPreviewSize(640, 640)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    
    cam.preview.link(xout.input)
    return pipeline

def main():
    create_trackbars()
    
    with dai.Device(create_pipeline()) as device:
        qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        
        while True:
            inVideo = qVideo.get()
            frame = inVideo.getCvFrame()
            
            mask, lower_bound, upper_bound, frame = process_frame(frame)
            
            cv2.putText(mask, f"Lower: {lower_bound}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            cv2.putText(mask, f"Upper: {upper_bound}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            
            cv2.imshow("Original", frame)
            cv2.imshow("Mask", mask)
            
            if cv2.waitKey(1) == ord('q'):
                break
    
    cv2.destroyAllWindows()
'''
def main():
    cap = cv2.VideoCapture(0)
    create_trackbars()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        mask, lower_bound, upper_bound, frame = process_frame(frame)
        
        cv2.putText(mask, f"Lower: {lower_bound}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        cv2.putText(mask, f"Upper: {upper_bound}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
'''
if __name__ == "__main__":
    main()

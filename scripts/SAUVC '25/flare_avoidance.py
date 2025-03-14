#!/usr/bin/env python3
import cv2
import rospy, time
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32
import depthai as dai
import numpy as np
from tvmc import MotionController, DoF, ControlMode
from PID_CONSTANTS import *

DATA_SOURCE = "sensors"

h_min = 0
h_max = 43
s_min = 0
s_max = 255
v_min = 0
v_max = 255
blur = 1
erode = 0
dilate = 5

current_yaw = None
motion = MotionController()

def process_frame(frame):
    global h_min, h_max, s_min, s_max, v_min, v_max, blur, erode, dilate
    
    frame = cv2.GaussianBlur(frame, (blur, blur), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=erode)
    mask = cv2.dilate(mask, kernel, iterations=dilate)
    
    return mask, lower_bound, upper_bound

def create_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    
    cam.setPreviewSize(640, 640)
    cam.setPreviewKeepAspectRatio(False)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(15)
    
    cam.preview.link(xout.input)
    return pipeline

def align_bot(target_x):
    global motion, current_yaw
    print(f"aligning {target_x}")
    yaw_target = (target_x - 320) * 0.06
    motion.set_target_point(DoF.YAW, current_yaw + yaw_target)
    time.sleep(0.5)

def avoid_flare(mask, current):
    global motion
    print("avoindance sequence")
    motion.set_target_point(DoF.YAW, (current + 90)%360)
    time.sleep(3)
    motion.set_thrust(DoF.SURGE, 1)
    time.sleep(2)
    motion.set_thrust(DoF.SURGE, 0)
    motion.set_target_point(DoF.YAW, current)
    time.sleep(3)
    motion.set_thrust(DoF.SURGE, 1)
    time.sleep(2)
    motion.set_thrust(DoF.SURGE, 0)


def main():
    

    with dai.Device(create_pipeline()) as device:
        qVideo = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        
        while True:
            inVideo = qVideo.get()
            frame = inVideo.getCvFrame()
            
            mask, lower_bound, upper_bound = process_frame(frame)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                target_x = x + w // 2
                align_bot(target_x)
                if abs(target_x - 320) < 20 and w * h > 5000:
                    avoid_flare(mask, current_yaw)
            
            cv2.putText(mask, f"Lower: {lower_bound}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            cv2.putText(mask, f"Upper: {upper_bound}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            
            cv2.imshow("Original", frame)
            cv2.imshow("Mask", mask)
            
            if cv2.waitKey(1) == ord('q'):
                break
    
    cv2.destroyAllWindows()

def on_orientation(vec: Vector3):
    global current_yaw, motion
    current_yaw = vec.z
    motion.set_current_point(DoF.YAW, vec.z)

def enable_pid():
    global motion
    motion.set_pid_constants(DoF.HEAVE, HEAVE_KP, HEAVE_KI, HEAVE_KD, HEAVE_ACCEPTABLE_ERROR, HEAVE_OFFSET)
    motion.set_pid_limits(DoF.HEAVE, -10, 10, -25, 25)
    motion.set_control_mode(DoF.HEAVE, ControlMode.CLOSED_LOOP)
    
    motion.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
    motion.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)

orientation_sub = rospy.Subscriber(f"/{DATA_SOURCE}/orientation", Vector3, on_orientation)

if __name__ == "__main__":

    motion.set_target_point(DoF.YAW, current_yaw)
    motion.set_target_point(DoF.HEAVE, 0.25)
    enable_pid()

    main()

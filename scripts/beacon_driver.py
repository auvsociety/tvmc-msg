#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from tvmc import MotionController, DoF, ControlMode

detections = []
frame = None
OFFSET = 50  # Acceptable error in pixels
FRAME_WIDTH = 640

PID_STATUS = {
    "HEAVE" : False,
    "PITCH" : False,
    "ROLL" : False,
    "YAW" : False
}

m = MotionController()
m.set_control_mode(DoF.YAW, ControlMode.OPEN_LOOP)



HEAVE_KP = -40
HEAVE_KI = 0.09
HEAVE_KD =  4.7
HEAVE_TARGET = 0.25
HEAVE_ACCEPTABLE_ERROR = 0.01
HEAVE_OFFSET = -0.11

PITCH_KP = -0.24/2
PITCH_KI = 0.0015
PITCH_KD = 0.2
PITCH_TARGET = 0
PITCH_ACCEPTABLE_ERROR = 0.7
PITCH_OFFSET = -0.5

ROLL_KP = 0.1
ROLL_KI = 0
ROLL_KD = 0.4
ROLL_TARGET = 0
ROLL_ACCEPTABLE_ERROR = 1.5

YAW_KP = 0
YAW_KI = 0
YAW_KD = 0
YAW_TARGET  = 65
YAW_ACCEPTABLE_ERROR = 1







if PID_STATUS["HEAVE"]:
    m.set_pid_constants(
        DoF.HEAVE,
        HEAVE_KP,
        HEAVE_KI,
        HEAVE_KD,
        HEAVE_ACCEPTABLE_ERROR,
        HEAVE_OFFSET,
    )
    m.set_pid_limits(DoF.HEAVE, -10, 10, -25, 25)
    m.set_target_point(DoF.HEAVE, HEAVE_TARGET)

if PID_STATUS["PITCH"]:
    m.set_pid_constants(
        DoF.PITCH, PITCH_KP, PITCH_KI, PITCH_KD, PITCH_ACCEPTABLE_ERROR
    )
    m.set_pid_limits(DoF.PITCH, -10, 10, -25, 25)
    m.set_target_point(DoF.PITCH, PITCH_TARGET)

if PID_STATUS["ROLL"]:
    m.set_pid_constants(DoF.ROLL, ROLL_KP, ROLL_KI, ROLL_KD, ROLL_ACCEPTABLE_ERROR)
    m.set_target_point(DoF.ROLL, ROLL_TARGET)

if PID_STATUS["YAW"]:
    m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
    





def detection_callback(msg):
    global detections
    detections = msg.data

def frame_callback(msg):
    global frame
    np_arr = np.frombuffer(msg.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame is not None:
        center_x = FRAME_WIDTH // 2
        target_x = None
        
        if detections:
            for i in range(0, len(detections), 5):
                xmin, ymin, xmax, ymax, confidence = detections[i:i+5]
                xmin, ymin, xmax, ymax = int(xmin * frame.shape[1]), int(ymin * frame.shape[0]), int(xmax * frame.shape[1]), int(ymax * frame.shape[0])
                target_x = (xmin + xmax) // 2
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Center: {target_x}", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if target_x is not None:
            error = target_x - center_x
            if abs(error) > OFFSET:
                yaw_adjustment = -0.1 * error
                m.set_target_point(DoF.YAW, yaw_adjustment)
                thrust = error**2 - -10
                m.set_thrust(DoF.YAW, thrust)
        
        cv2.imshow("YOLO Yaw Control", frame)
        cv2.waitKey(1)

def main():
    rospy.init_node('oakd_yolo_yaw_controller', anonymous=True)
    rospy.Subscriber('/yolo_detections', Float32MultiArray, detection_callback)
    rospy.Subscriber('/yolo_frame', CompressedImage, frame_callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

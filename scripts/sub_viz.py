#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

detections = []
frame = None

def detection_callback(msg):
    global detections
    detections = msg.data

def frame_callback(msg):
    global frame
    np_arr = np.frombuffer(msg.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame is not None and detections:
        for i in range(0, len(detections), 5):
            xmin, ymin, xmax, ymax, confidence = detections[i:i+5]
            xmin, ymin, xmax, ymax = int(xmin * frame.shape[1]), int(ymin * frame.shape[0]), int(xmax * frame.shape[1]), int(ymax * frame.shape[0])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("YOLO Detections", frame)
        cv2.waitKey(1)

def main():
    rospy.init_node('oakd_yolo_subscriber', anonymous=True)
    rospy.Subscriber('/yolo_detections', Float32MultiArray, detection_callback)
    rospy.Subscriber('/yolo_frame', CompressedImage, frame_callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

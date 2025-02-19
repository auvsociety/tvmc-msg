#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray

def callback(msg):
    rospy.loginfo("Received detections:")
    detections = msg.data
    for i in range(0, len(detections), 5):
        bbox = detections[i:i+5]
        rospy.loginfo(f"BBox: xmin={bbox[0]}, ymin={bbox[1]}, xmax={bbox[2]}, ymax={bbox[3]}, confidence={bbox[4]}")

def main():
    rospy.init_node('oakd_yolo_subscriber', anonymous=True)
    rospy.Subscriber('/yolo_detections', Float32MultiArray, callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
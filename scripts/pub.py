#!/usr/bin/env python3
from pathlib import Path
import rospy
import cv2
import depthai as dai
import numpy as np
from std_msgs.msg import Float32MultiArray

# Initialize ROS node
def main():
    rospy.init_node('oakd_yolo_publisher', anonymous=True)
    pub = rospy.Publisher('/yolo_detections', Float32MultiArray, queue_size=10)
    
    nnPath = str((Path(__file__).parent / Path('./../yolo_models/beacon.blob')).resolve().absolute())

    labelMap = ["Beacon"]
    
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    nn = pipeline.create(dai.node.YoloDetectionNetwork)
    nnout = pipeline.create(dai.node.XLinkOut)
    
    nnout.setStreamName("nn")
    cam.setPreviewSize(640, 640)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(15)
    
    nn.setConfidenceThreshold(0.5)
    nn.setNumClasses(1)
    nn.setCoordinateSize(4)
    nn.setAnchors([
                    10.0,
                    13.0,
                    16.0,
                    30.0,
                    33.0,
                    23.0,
                    30.0,
                    61.0,
                    62.0,
                    45.0,
                    59.0,
                    119.0,
                    116.0,
                    90.0,
                    156.0,
                    198.0,
                    373.0,
                    326.0
                ])
    nn.setAnchorMasks({
                    "side80": [
                        0,
                        1,
                        2
                    ],
                    "side40": [
                        3,
                        4,
                        5
                    ],
                    "side20": [
                        6,
                        7,
                        8
                    ]
                })
    nn.setIouThreshold(0.5)
    nn.setBlobPath(nnPath)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    
    cam.preview.link(nn.input)
    nn.out.link(nnout.input)
    
    with dai.Device(pipeline) as device:
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        
        while not rospy.is_shutdown():
            inDet = qDet.get()
            if inDet is not None:
                detections = inDet.detections
                msg = Float32MultiArray()
                for det in detections:
                    msg.data.extend([det.xmin, det.ymin, det.xmax, det.ymax, det.confidence])
                pub.publish(msg)
            rospy.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
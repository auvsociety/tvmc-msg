#!/usr/bin/env python3
from pathlib import Path
import rospy, time
import cv2
import depthai as dai
import numpy as np
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

def create_pipeline():
    pipeline = dai.Pipeline()
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    nn = pipeline.create(dai.node.YoloDetectionNetwork)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xin_nn = pipeline.create(dai.node.XLinkIn)
    nnout = pipeline.create(dai.node.XLinkOut)
    
    xout_depth.setStreamName("depth")
    xin_nn.setStreamName("depth_to_nn")
    nnout.setStreamName("nn")
    
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")
    
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setSubpixel(False)
    
    nn.setConfidenceThreshold(0.4)
    nn.setNumClasses(1)
    nn.setCoordinateSize(4)
    nn.setAnchors([10.0,13.0,16.0,30.0,33.0,23.0,30.0,61.0,62.0,45.0,59.0,119.0,116.0,90.0,156.0,198.0,373.0,326.0])
    nn.setAnchorMasks({"side80": [0,1,2], "side40": [3,4,5], "side20": [6,7,8]})
    nn.setIouThreshold(0.5)
    nnPath = str((Path(__file__).parent / Path('./../../yolo_models/gate.blob')).resolve().absolute())
    nn.setBlobPath(nnPath)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.depth.link(xout_depth.input)
    xin_nn.out.link(nn.input)
    nn.out.link(nnout.input)
    
    return pipeline

def main():
    rospy.init_node('oakd_yolo_publisher', anonymous=True)
    pub_detections = rospy.Publisher('/yolo_detections', Float32MultiArray, queue_size=10)
    pub_depth = rospy.Publisher('/yolo_frame', CompressedImage, queue_size=10)
    
    with dai.Device(create_pipeline()) as device:
        print("device started")
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        qDepthToNN = device.getInputQueue(name="depth_to_nn",maxSize = 4, blocking = False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        
        while 1:
            inDepth = qDepth.tryGet()
            inDet = qDet.tryGet()
            print(".")
            if inDepth is not None:
                depthFrame = inDepth.getFrame()
                depthFrame = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depthFrame = cv2.resize(depthFrame, (640, 640))
                cv2.imshow("data", depthFrame)
                
                
                # msg_depth = CompressedImage()
                # msg_depth.header.stamp = rospy.Time.now()
                # msg_depth.format = "jpeg"
                # msg_depth.data = np.array(cv2.imencode('.jpg', depthFrame)[1]).tobytes()
                # pub_depth.publish(msg_depth)
                
                dai_frame = dai.ImgFrame()
                dai_frame.setData(depthFrame.tobytes())
                dai_frame.setWidth(640)
                dai_frame.setHeight(640)
                dai_frame.setType(dai.ImgFrame.Type.GRAY8)
                qDepthToNN.send(dai_frame)
            
            if inDet is not None:
                detections = inDet.detections
                msg = Float32MultiArray()
                for det in detections:
                    msg.data.extend([det.xmin, det.ymin, det.xmax, det.ymax, det.confidence])
                pub_detections.publish(msg)
            if cv2.waitKey(1) == ord('q'):
                break
            rospy.sleep(0.1)
        print("outside while")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

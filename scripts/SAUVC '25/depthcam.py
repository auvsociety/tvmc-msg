#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np

def create_pipeline():
    pipeline = dai.Pipeline()
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    xout = pipeline.create(dai.node.XLinkOut)

    xout.setStreamName("depth")

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setSubpixel(False)

    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.depth.link(xout.input)

    return pipeline


with dai.Device(create_pipeline()) as device:
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    while True:
        inDepth = qDepth.get()
        depthFrame = inDepth.getFrame()
        depthFrame = cv2.normalize(depthFrame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depthFrame = cv2.resize(depthFrame, (640,640))
        depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)
        cv2.imshow("Depth with Quadrilateral Detection", depthFrame)
        if cv2.waitKey(1) == ord('q'):
            break



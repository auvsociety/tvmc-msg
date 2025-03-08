#!/usr/bin/env python3
import cv2
import depthai as dai
import time
import numpy as np

nnPath = './../yolo_models/gate.blob'

labelMap = [
	"Beacon"
]
syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
cam = pipeline.create(dai.node.ColorCamera)
nn = pipeline.create(dai.node.YoloDetectionNetwork)
xout= pipeline.create(dai.node.XLinkOut)
nnout = pipeline.create(dai.node.XLinkOut)

xout.setStreamName("rgb")
nnout.setStreamName("nn")

# Properties
cam.setPreviewSize(640,640)
cam.setPreviewKeepAspectRatio(False)
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
nn.setIouThreshold(0.7)
nn.setBlobPath(nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)



cam.preview.link(nn.input)
if syncNN:
    nn.passthrough.link(xout.input)
else:
    cam.preview.link(xout.input)

nn.out.link(nnout.input)



# Connect to device and start pipeline
with dai.Device(pipeline) as device:


    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    
    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    while True:
        
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break



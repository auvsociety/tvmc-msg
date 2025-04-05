import cv2
import numpy as np
import math

class Line:
    def __init__(self, x1, y1, x2, y2) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __repr__(self):
        return f"Line({self.x1}, {self.y1}, {self.x2}, {self.y2})"

    def x_pos(self):
        return np.mean([self.x1, self.x2])

    def y_pos(self):
        return np.mean([self.y1, self.y2])

    def bottom(self):
        return max(self.y1, self.y2)

    def length(self):
        return np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    def angle(self):
        angle = math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))
        return angle if angle >= 0 else 180 + angle
    
cam = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
while cv2.waitKey(1) != ord('q'):
    image = cam.read()[1]
    image = cv2.resize(image, (640, 360))
    image = cv2.GaussianBlur(image, (3, 3), 1)
    # hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Camer",opening)
    opening = 255 - opening
    opening = cv2.erode(opening, None, iterations=1)
    minLineLength = 50
    lines = cv2.HoughLinesP(
        image=opening,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        lines=np.array([]),
        minLineLength=minLineLength,
        maxLineGap=5,
    )
    vertical_lines = []

    if lines is not None:
        for line in lines:
            line = Line(*line[0])

            # print(theta)
            angle_threshold = 20

            if abs(90 - line.angle()) <= angle_threshold:
                cv2.line(
                    image,
                    (line.x1, line.y1),
                    (line.x2, line.y2),
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
                vertical_lines.append(line)

    # bunch lines together based on x position
    vertical_lines.sort(key=lambda x: x.x_pos())

    # cluster lines
    clusters = []
    cluster = []



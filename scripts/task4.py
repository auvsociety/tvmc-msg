#!/usr/bin/env python3
import math
import time
import threading

import cv2
import rospy
import cv_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from statemachine import StateMachine, State
import numpy as np
from tvmc import MotionController, DoF, ControlMode
from statemachine.contrib.diagram import DotGraphMachine


# globals
CURRENT_YAW = 0
START_YAW = 0
REVERSE_YAW = -1
DATA_SOURCE = "emulation"
IMAGE_ROSTOPIC ="/oak/rgb/image_raw"

# H_FOV = 72.14
H_FOV = 62.14

HEAVE_KP = -170
HEAVE_KI = -10
HEAVE_KD = -60
HEAVE_TARGET = 1.2
HEAVE_ACCEPTABLE_ERROR = 0.025
HEAVE_OFFSET = 8

PITCH_KP = 10
PITCH_KI = 0
PITCH_KD = 4
PITCH_TARGET = 0
PITCH_ACCEPTABLE_ERROR = 1.5

ROLL_KP = 1
ROLL_KI = 0
ROLL_KD = 0.4
ROLL_TARGET = 0
ROLL_ACCEPTABLE_ERROR = 1.5

YAW_KP = 0.86
# YAW_KP = 1.6
YAW_KI = 0
YAW_KD = 0.3
YAW_TARGET = 45
YAW_ACCEPTABLE_ERROR = 1.5


FRAME_WIDTH = 640
GATE_ACCEPTABLE_ERROR = 5

SWAY_THRUST = 50

def predict_depth(line1, line2):
    fov_w, fov_h = 62 * math.pi / 180, 46 * math.pi / 180
    px_W, px_H = 384, 216
    # print(line1, line2)
    # print(np.subtract(line1[0, :], line1[1,:]))
    l1 = np.sqrt(np.sum(np.square(np.subtract(line1[0, :], line1[1, :])), axis=0))
    l2 = np.sqrt(np.sum(np.square(np.subtract(line2[0, :], line2[1, :])), axis=0))
    # print(l1, l2)
    if abs(l2 - l1) / l2 < 0.3:
        l = (l1 + l2) / 2
    elif abs(l2 - l1) / l2 < 0.5:
        l = max(l1, l2)
    else:
        # not pole
        return -1
    # real length of pole in metres
    real_l = 1.5
    ppm = l / real_l
    H = px_H / ppm
    depth = H / (2 * math.tan(fov_h / 2))
    # print(depth)
    return depth
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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


class Memory:
    def __init__(
        self,
        x,
        y,
        line_x1=None,
        line_x2=None,
        min_consistency=7,
        history=14,
        max_deviation=200,
    ):
        self.x = x
        self.y = y
        self.line_x1 = line_x1
        self.line_x2 = line_x2
        self.history = history
        self.consistency = [0] * self.history
        self.consistency[-1] = 1
        self.min_consistency = min_consistency
        self.max_deviation = max_deviation

    def __repr__(self):
        return f"Memory({self.x}, {self.y}, {self.consistency})"

    def run_frame(self):
        self.consistency.append(0)
        self.consistency = self.consistency[-self.history :]
        return sum(self.consistency)

    def update(self, x, y, line_x1=None, line_x2=None):
        dev = abs(self.x - x)

        if dev > self.max_deviation:
            return False

        self.x, self.y = x, y
        self.line_x1, self.line_x2 = line_x1, line_x2
        self.consistency[-1] = 1
        return True

    def recall(self):
        if sum(self.consistency) >= self.min_consistency:
            return self.x, self.y

        return None


class Memories:
    def __init__(self, **memory_params):
        self.memories = []
        self.params = memory_params

    def __repr__(self):
        return f"Memories({self.memories})"

    def run_frame(self):
        new_memories = []

        for memory in self.memories:
            if memory.run_frame():
                new_memories.append(memory)

        self.memories = new_memories

    def remember(self, x, y, line_x1=None, line_x2=None):
        remembered = False

        for memory in self.memories:
            if memory.update(x, y, line_x1, line_x2):
                remembered = True
                break

        if not remembered:
            self.memories.append(Memory(x, y, line_x1, line_x2, **self.params))

    def recall(self):
        memories = [memory.recall() for memory in self.memories]
        return [*filter(lambda x: x is not None, memories)]

    def best_recall(self, consistency=5):
        memories = []

        for memory in self.memories:
            if sum(memory.consistency) >= consistency:
                memories.append((memory, sum(memory.consistency)))

        if memories:
            return max(memories, key=lambda x: x[1])[0]

        return None

memories = Memories()
clustered_lines = []

def detect_gate(image):
    global clustered_lines

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
    opening = 255 - opening
    opening = cv2.erode(opening, None, iterations=1)

    # canny = cv2.Canny(
    #     cv2.GaussianBlur(clahe.apply(gray), (11, 11), 0),
    #     300,
    #     700,
    #     apertureSize=5,
    # )

    # canny = cv2.dilate(canny, None, iterations=1)
    # cv2.imshow("canny", canny)

    cv2.imshow("opening", opening)

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

    for i in range(len(vertical_lines) - 1):
        cluster.append(vertical_lines[i])
        if vertical_lines[i + 1].x_pos() - vertical_lines[i].x_pos() > 20:
            clusters.append(cluster)
            cluster = []

    if cluster:
        clusters.append(cluster)

    clustered_lines = []

    for cluster in filter(lambda x: len(x) > 1, clusters):
        ys = [x.y1 for x in cluster] + [x.y2 for x in cluster]
        y1 = min(ys)
        y2 = max(ys)

        x = int(np.mean([x.x_pos() for x in cluster]))
        cv2.line(image, (x, y1), (x, y2), (0, 0, 255), 3, cv2.LINE_AA)
        clustered_line = Line(x, y1, x, y2)

        if clustered_line.length() > 50:
            clustered_lines.append(clustered_line)

    memories.run_frame()

    clustered_lines_copy = clustered_lines[:]

    while clustered_lines:
        line = clustered_lines.pop()

        for other_line in clustered_lines:
            if abs(line.y_pos() - other_line.y_pos()) < 100:
            # if abs(line.bottom() - other_line.bottom()) < 25:
                gate = np.mean(
                    [
                        [line.x_pos(), other_line.x_pos()],
                        [line.y_pos(), other_line.y_pos()],
                    ],
                    axis=1,
                )
                memories.remember(
                    gate[0],
                    gate[1],
                    min(line.x_pos(), other_line.x_pos()),
                    max(line.x_pos(), other_line.x_pos()),
                )
    
    clustered_lines = clustered_lines_copy

    for gate in memories.recall():
        cv2.circle(image, tuple(int(x) for x in gate), 10, (0, 255, 255), -1)

    cv2.imshow("image", image)
    cv2.waitKey(1)
def set_yaw(angle):
    pass


def stop_yaw():
    pass


def surge(thrust):
    pass


class FlareTask(StateMachine):
    # initializing_camera = State()
    # fixing_yaw = State()
    # heaving_down = State()
    # surging_forward = State()

    gatePass = State(initial=True)
    aboutTurn = State()
    swaySide = State()
    scanApril = State()
    centerApril = State()
    turn45 = State()
    turn90 = State()
    turn135 = State()
    swaySearchFlare = State()
    hitVisible1 = State()
    hitVisible2 = State()
    hitVisible3 = State()
    # flareCorrection = State()
    # surging = State()
    finished = State(final=True)

    begin = gatePass.to(aboutTurn)
    startSway = aboutTurn.to(swaySide)
    recheck = swaySearchFlare.to(turn45)
    startFlareCheck = centerApril.to(turn45)
    # finish = gatePass.to(finished)
    proceed = (
        swaySide.to(scanApril,cond="detectedApril")
        | swaySide.to(swaySide,unless="detectedApril",internal=True)
        | scanApril.to(centerApril,cond="readApril")
        | scanApril.to(scanApril,unless="readApril")
        | turn45.to(turn90, cond="turn1")
        | turn90.to(turn135,cond="turn2")
        | turn135.to(hitVisible1,unless="moreToFind")
        | turn135.to(swaySearchFlare,cond="swayToSearchCondn")
        | turn135.to(hitVisible2,cond = "foundTwo")
        | turn135.to(hitVisible3,cond = "foundOne")
        | hitVisible1.to(hitVisible2,cond="hit1")
        | hitVisible2.to(hitVisible3,cond="hit2")
        | hitVisible3.to(finished,cond="hit3")

    )

    def on_enter_gatePass(self):
        # Init state
        print("Task 4 sequence initiated")
        time.sleep(1) #let bot relax
        print("Initializing Sensors.")
        self.image_sub = rospy.Subscriber(IMAGE_ROSTOPIC, Image, self.on_image)
        print("Camera done.")
        self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
        self.orientation_sub = rospy.Subscriber(
            f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation
        )
        self.begin() #initiate sequence

    def on_enter_aboutTurn(self):
        print("ABOUT TURN TIMEE")
        self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
        self.orientation_sub = rospy.Subscriber(
            f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation
        )
        while self.yaw_lock is None:
            time.sleep(0.1)
        self.set_yaw(self.current_yaw + 180)
        global YAW_TARGET
        YAW_TARGET = self.current_yaw+180
        self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)
        while (
            self.current_yaw is None
            or abs(self.current_yaw - YAW_TARGET) > YAW_ACCEPTABLE_ERROR * 3
        ):
            time.sleep(0.1)
        print("Rotated 180")
        self.startSway()
    
    def on_enter_swaySide(self):
        global SWAY_THRUST
        self.m.set_control_mode(DoF.SWAY,ControlMode.OPEN_LOOP)
        self.m.set_thrust(DoF.SWAY,SWAY_THRUST)
        #Wait for April tag to be detected, set the variable when it is detected
        while (not self.detectedApril):
            time.sleep(0.1)
        
    
    def on_exit_swaySide(self):
        self.m.set_thrust(DoF.SWAY,0)

    def on_enter_scanApril(self):
        print("April tag detected, reading it")
        #TODO Read APRIL TAG
        print("April Tag read")
        ...
     
    def on_exit_scanApril(self):
        ...

    def on_enter_centerApril(self):
        global SWAY_THRUST
        self.m.control_mode(DoF.SWAY,ControlMode.OPEN_LOOP)
        self.m.set_thrust(DoF.SWAY,SWAY_THRUST)
        while(not self.centeredApril):
            time.sleep(0.1)
        self.startFlareCheck()
        ...
    
    def on_exit_centerApril(self):
        self.m.set_thrust(DoF.SWAY,0)
        ...
    
    def on_enter_turn45(self):
        print("45 degree scan turn")
        self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
        self.orientation_sub = rospy.Subscriber(
            f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation
        )
        while self.yaw_lock is None:
            time.sleep(0.1)
        global YAW_TARGET
        self.set_yaw(YAW_TARGET - 135)
        self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)
        while (
            self.current_yaw is None
            or abs(self.current_yaw - YAW_TARGET) > YAW_ACCEPTABLE_ERROR * 3
        ):
            time.sleep(0.1)
        print("Scan 1 turn done")
        self.turn1 = True
    
    def on_enter_turn90(self):
        print("90 Degree scan turn")
        global YAW_TARGET
        
        self.set_yaw(YAW_TARGET-180)
        while (
            self.current_yaw is None 
            or abs(self.current_yaw - YAW_TARGET) > YAW_ACCEPTABLE_ERROR * 3
        ):
            time.sleep(0.1)
        self.turn2=True
    def on_enter_turn135(self):
        print("90 Degree scan turn")
        global YAW_TARGET
        
        self.set_yaw(YAW_TARGET-180-45)
        while (
            self.current_yaw is None 
            or abs(self.current_yaw - YAW_TARGET) > YAW_ACCEPTABLE_ERROR * 3
        ):
            time.sleep(0.1)
        self.turn3=True
    def swayable(self):
        # returns true if we didnt sway search already
        return self.swayCount == 0 
    def moreToFind(self):
        #Will return true if we didnt find 3 flares
        return not self.flareCount == 3
    def swayToSearchCondn(self):
        # Sway to search if we didnt sway already and we didnt find 3 flares
        return self.moreToFind() and self.swayable()
    def foundOne(self):
        # condition for when we found only one flare even after alot of bs (swaying)
        return (not self.swayable()) and self.flareCount == 1
    def foundTwo(self):
        # condition for when we found only one flare even after alot of bs (swaying)
        return (not self.swayable()) and self.flareCount == 2


    def on_enter_swaySearchSway(self):
        global SWAY_THRUST
        self.m.control_mode(DoF.SWAY,ControlMode.OPEN_LOOP)
        self.m.set_thrust(DoF.SWAY,SWAY_THRUST)
        self.timer = threading.Thread(target=self.blind_timer_async,daemon=True)
    def on_exit_swaySearchSway(self):
        self.m.set_thrust(DoF.SWAY,0)
        self.swayCount +=  1
    
    # def on_enter_surging(self):
    #     print("Surging forward to flare.")
    #     self.enable_pitch_pid()
    #     self.m.set_thrust(DoF.SURGE, 40)

    # def on_exit_surging(self):
    #     print("Stopping Surge")
    #     self.disable_pitch_pid()
    #     self.m.set_thrust(DoF.SURGE, 0)

    
    def correct_towards_flare_async(self):
        self.correcting = True
        self.set_yaw((self.correction_required() + self.current_yaw) / 2)

        time_slept = 0
        # direction = 'left' if self.correction_required() < self.current_yaw else 'right'
        # self.m.set_thrust(DoF.SWAY, 25 * (1 if direction == 'left' else -1))
        # CHANGE self.correction_required() function
        while self.correction_required() and time_slept < 0.5:
            time.sleep(0.1)
            time_slept += 0.1
            # self.set_yaw(self.correction_required())

        # self.m.set_thrust(DoF.SWAY, 0)
        self.correction_thread = None
        self.correcting = False
        self.proceed()


    # def on_enter_flareCorrection(self):
    #     print("Attempting to correcting heading towards gate.")
    #     # self.set_yaw((self.correction_required() + self.current_yaw) / 2)

    #     if not self.correction_thread:
    #         self.correction_thread = threading.Thread(target=self.correct_towards_flare_async, daemon=True)
    #         self.correction_thread.start()


    def on_enter_hitVisible1(self):

        self.hit1 = True
        ...
    def on_enter_hitVisible1(self):

        self.hit2 = True
        ...
    def on_enter_hitVisible1(self):

        self.hit3=  True
        ...

    

    def blind_timer_async(self):
        time.sleep(2)
        self.swayCount += 1
        self.recheck()

    def __init__(self):
        self.m = MotionController()

        self.image_sub = None
        self.orientation_sub = None
        self.depth_sub = None

        self.bridge = cv_bridge.CvBridge()

        self.current_depth = None
        self.current_yaw = None
        self.yaw_lock = None

        self.camera_ready = False
        self.gate_visible = False

        self.correction_thread = None
        self.single_pole_warning = False
        self.FOV_exceeded = 0
        self.blind_timer = None
        self.correcting = False

        self.detectedApril = False
        self.readApril = False 
        self.swayDir = 0 
        self.centeredApril = False

        self.turn1 = False
        self.turn2 = False
        self.turn3 = False
        self.flareCount = 0
        self.swayCount = 0
        self.flareCentered = False
        self.hitCount = 0
        self.hit1 = False
        self.hit2 = False
        self.hit3 = False
        # Direction to sway to center April Tag, 1 for left, -1 for right, set after getting angle from CV 
        super(FlareTask, self).__init__()



    def on_depth(self, depth: Float32):
        self.current_depth = depth.data
        self.m.set_current_point(DoF.HEAVE, depth.data)

    def set_yaw(self, angle):
        self.yaw_lock = angle
        self.m.set_target_point(DoF.YAW, angle)

    def on_orientation(self, vec: Vector3):
        self.current_yaw = vec.z

        if not self.yaw_lock:
            self.yaw_lock = vec.z

        self.m.set_current_point(DoF.YAW, vec.z)

        # print(f"\rYaw: {self.current_yaw}, Lock: {self.yaw_lock}", end='')






    def enable_pitch_pid(self):
        self.m.set_pid_constants(
            DoF.PITCH, PITCH_KP, PITCH_KI, PITCH_KD, PITCH_ACCEPTABLE_ERROR
        )
        # self.m.set_pid_limits(DoF.PITCH, -10, 10, -25, 25)
        self.m.set_target_point(DoF.PITCH, PITCH_TARGET)
        self.m.set_control_mode(DoF.PITCH, ControlMode.CLOSED_LOOP)

    def disable_pitch_pid(self):
        self.m.set_control_mode(DoF.PITCH, ControlMode.OPEN_LOOP)


    
    def on_enter_finished(self):
        print("Qualified.")
        self.m.set_target_point(DoF.HEAVE, 0)
        self.set_yaw(self.current_yaw + 180 % 180)




    # def on_correct_for_gate(self):
    #     new_heading = (self.current_yaw + self.correction_required()) * 10
    #     print(f"Correcting yaw heading to {new_heading} from {self.current_yaw}.")
    #     set_yaw(new_heading)

    def correction_required(self):
        gate = memories.best_recall()
        
        if not gate:
            return False
        
        x_pos = gate.x

        deviation = x_pos - FRAME_WIDTH / 2
        yaw_required = deviation / FRAME_WIDTH * H_FOV

        if abs(yaw_required) > GATE_ACCEPTABLE_ERROR:
            return REVERSE_YAW * yaw_required + self.current_yaw

        return False

    
    def on_image(self, image):
        image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        detect_gate(image)

        if not self.camera_ready or self.finished.is_active:
            return

        if memories.best_recall():
            self.gate_visible = True
            self.check_FOV_exceeded()

            if self.correction_required():
                # self.set_yaw(self.correction_required())
                self.correct_for_gate()
            elif not self.correcting:
                self.surge()
        else:
            self.preventive_measures()
            self.gate_visible = False
            
            if not self.correcting:
                self.surge()
if __name__ == "__main__":
    # rospy.init_node("Node")
    # task = FlareTask()
    
    graph = DotGraphMachine(FlareTask)()
    graph.write_png("/home/sushi/flare.png")
    
    # rospy.spin()

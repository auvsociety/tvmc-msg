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


# globals
CURRENT_YAW = 0
START_YAW = 0

H_FOV = 71.4

HEAVE_KP = -170
HEAVE_KI = -10
HEAVE_KD = -60
HEAVE_TARGET = 0.8
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
YAW_KI = 0
YAW_KD = 0.3
YAW_TARGET = 45
YAW_ACCEPTABLE_ERROR = 1.5


FRAME_WIDTH = 640
GATE_ACCEPTABLE_ERROR = 5


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
            if abs(line.bottom() - other_line.bottom()) < 25:
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


class QualificationTask(StateMachine):
    wait_to_start = State(initial=True)

    initializing_camera = State()

    fixing_yaw = State()
    heaving_down = State()

    surging_forward = State()

    correcting_for_gate = State()
    surging_towards_gate = State()

    blind_surge = State()
    finished = State(final=True)

    start_task = wait_to_start.to(initializing_camera)
    fix_yaw = initializing_camera.to(fixing_yaw)
    heave_down = fixing_yaw.to(heaving_down)
    surge_forward = heaving_down.to(surging_forward)

    correct_for_gate = (
        correcting_for_gate.to(correcting_for_gate, internal=True)
        | surging_forward.to(correcting_for_gate)
        | surging_towards_gate.to(correcting_for_gate)
    )

    surge_towards_gate = (
        surging_towards_gate.to(surging_towards_gate, internal=True)
        | correcting_for_gate.to(surging_towards_gate)
        | surging_forward.to(surging_towards_gate)
    )

    surge = (
        surging_forward.to(blind_surge, cond="gate_about_to_pass")
        | surging_towards_gate.to(blind_surge, cond="gate_about_to_pass")
        | blind_surge.to(blind_surge, cond="gate_about_to_pass", internal=True)
        | correcting_for_gate.to(surging_towards_gate, cond="gate_visible")
        | correcting_for_gate.to(surging_forward, unless="gate_visible")
        | surging_towards_gate.to(
            surging_towards_gate, cond="gate_visible", internal=True
        )
        | surging_towards_gate.to(surging_forward, unless="gate_visible")
        | surging_forward.to(surging_towards_gate, cond="gate_visible")
        | surging_forward.to(surging_forward, unless="gate_visible", internal=True)
    )

    blindly_surge = surging_towards_gate.to(blind_surge)

    finish = blind_surge.to(finished)

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
        self.singe_pole_warning = False
        self.FOV_exceeded = 0
        self.blind_timer = None

        super(QualificationTask, self).__init__()

    def on_enter_wait_to_start(self):
        # wait for some time for everything to be fine
        print("Waiting to start.")
        self.m.start()
        time.sleep(3)
        self.start_task()

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

    def on_enter_initializing_camera(self):
        print("Initializing Sensors.")
        self.image_sub = rospy.Subscriber(
            "/emulation/camera/front/image_color", Image, self.on_image
        )
        print("Camera done.")

        self.fix_yaw()

    def on_enter_fixing_yaw(self):
        print("Attempting to fix yaw.")
        self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
        self.orientation_sub = rospy.Subscriber(
            "/emulation/orientation", Vector3, self.on_orientation
        )

        while not self.yaw_lock:
            time.sleep(0.1)

        self.set_yaw(self.yaw_lock)
        self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)
        print("Orientation done. Locked Yaw.")

        self.heave_down()

    def on_enter_heaving_down(self):
        print("Attempting to maintain depth.")

        self.m.set_pid_constants(
            DoF.HEAVE,
            HEAVE_KP,
            HEAVE_KI,
            HEAVE_KD,
            HEAVE_ACCEPTABLE_ERROR,
            HEAVE_OFFSET,
        )
        self.m.set_pid_limits(DoF.HEAVE, -10, 10, -25, 25)
        self.m.set_target_point(DoF.HEAVE, HEAVE_TARGET)
        self.depth_sub = rospy.Subscriber("/emulation/depth", Float32, self.on_depth)
        self.m.set_control_mode(DoF.HEAVE, ControlMode.CLOSED_LOOP)

        while (
            self.current_depth is None
            or abs(self.current_depth - HEAVE_TARGET) > HEAVE_ACCEPTABLE_ERROR * 3
        ):
            time.sleep(0.1)

        print("Heave done. Locked Depth.")

        self.camera_ready = True
        self.surge_forward()

    def enable_pitch_pid(self):
        self.m.set_pid_constants(
            DoF.PITCH, PITCH_KP, PITCH_KI, PITCH_KD, PITCH_ACCEPTABLE_ERROR
        )
        # self.m.set_pid_limits(DoF.PITCH, -10, 10, -25, 25)
        self.m.set_target_point(DoF.PITCH, PITCH_TARGET)
        self.m.set_control_mode(DoF.PITCH, ControlMode.CLOSED_LOOP)

    def disable_pitch_pid(self):
        self.m.set_control_mode(DoF.PITCH, ControlMode.OPEN_LOOP)

    def on_enter_surging_forward(self):
        print("Surging forward, looking for gate.")
        self.enable_pitch_pid()
        self.m.set_thrust(DoF.SURGE, 40)

    def on_exit_surging_forward(self):
        print("Stopping Surge.")
        self.disable_pitch_pid()
        self.m.set_thrust(DoF.SURGE, 0)

    def on_enter_surging_towards_gate(self):
        print("Surging towards gate.")
        self.enable_pitch_pid()
        self.m.set_thrust(DoF.SURGE, 30)

    def on_exit_surging_towards_gate(self):
        # print("Gate out of view.")
        self.disable_pitch_pid()
        self.m.set_thrust(DoF.SURGE, 0)

    def blind_timer_async(self):
        time.sleep(10)
        self.finish()

    def on_enter_blind_surge(self):
        print("About to pass through gate, going blind.")
        self.enable_pitch_pid()
        self.m.set_thrust(DoF.SURGE, 60)

        if not self.blind_timer:
            self.blind_timer = threading.Thread(target=self.blind_timer_async, daemon=True)
            self.blind_timer.start()

    def on_exit_blind_surge(self):
        print("Stopping Surge.")
        self.disable_pitch_pid()
        self.m.set_thrust(DoF.SURGE, 0)
    
    def on_enter_finished(self):
        print("Qualified.")
        self.m.set_target_point(DoF.HEAVE, 0)
        self.set_yaw(self.current_yaw + 180 % 180)

    def correct_towards_gate_async(self):
        self.set_yaw(self.correction_required())

        time_slept = 0

        while self.correction_required() and time_slept < 1:
            time.sleep(0.1)
            time_slept += 1
            self.set_yaw(self.correction_required())

        self.correction_thread = None
        self.surge()

    def on_enter_correcting_for_gate(self):
        print("Attempting to correcting heading towards gate.")
        # if not self.correction_thread:
        #     self.correction_thread = threading.Thread(target=self.correct_towards_gate_async, daemon=True)
        #     self.correction_thread.start()

    # def on_correct_for_gate(self):
    #     new_heading = (self.current_yaw + self.correction_required()) * 10
    #     print(f"Correcting yaw heading to {new_heading} from {self.current_yaw}.")
    #     set_yaw(new_heading)

    def correction_required(self):
        gate = memories.best_recall()

        x_pos = gate.x

        deviation = x_pos - FRAME_WIDTH / 2
        yaw_required = deviation / FRAME_WIDTH * H_FOV

        if abs(yaw_required) > GATE_ACCEPTABLE_ERROR:
            return yaw_required + self.current_yaw

        return False

    def preventive_measures(self):
        if len(clustered_lines) > 0 and not self.singe_pole_warning:
            print("detected sole line")
            if not self.singe_pole_warning:
                print("Detected sole line. Turning towards pole.")
                self.singe_pole_warning = True

            x_pos = clustered_lines[0].x_pos()

            deviation = x_pos - FRAME_WIDTH / 2
            yaw_required = deviation / FRAME_WIDTH * H_FOV

            if abs(yaw_required) > GATE_ACCEPTABLE_ERROR:
                self.set_yaw(yaw_required + self.current_yaw)

    def check_FOV_exceeded(self):
        gate = memories.best_recall()

        if not (gate.line_x1 and gate.line_x2):
            return

        if gate.line_x1 < FRAME_WIDTH * 0.2 and gate.line_x2 > FRAME_WIDTH * 0.8:
            self.FOV_exceeded += 1

    def gate_about_to_pass(self):
        return self.FOV_exceeded > 20

    def on_image(self, image):
        image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        detect_gate(image)

        if not self.camera_ready or self.finished.is_active:
            return

        if memories.best_recall():
            self.gate_visible = True
            self.check_FOV_exceeded()

            if self.correction_required():
                self.set_yaw(self.correction_required())
                self.correct_for_gate()
            else:
                self.surge()
        else:
            self.preventive_measures()
            self.gate_visible = False
            self.surge()


if __name__ == "__main__":
    # rospy.init_node("Node")
    task = QualificationTask()
    rospy.spin()

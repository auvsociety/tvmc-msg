#!/usr/bin/env python3
import rospy
import cv2, threading, time
import numpy as np
from statemachine import StateMachine, State
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32, Vector3
from tvmc import MotionController, DoF, ControlMode
from rose_tvmc_msg.msg import LEDControl


detections = []
frame = None
FRAME_WIDTH = 640
OFFSET = 40 / FRAME_WIDTH
RESTING_YAW_THRUST = 0
YAW_THRUST = 40
YAW_ADJUSTMENT_THRUST = None
YAW_TIME = 0.3
YAW_ADJUSTMENT_TIME = None
DATA_SOURCE = "sensors"
LEFT = None

PID_STATUS = {
    "HEAVE": False,
    "PITCH": False,
    "ROLL": False,
    "YAW": False
}


HEAVE_TARGET_OFFSET = -0.07
HEAVE_KP = -45
HEAVE_KI = -0.05
HEAVE_KD =  25
HEAVE_TARGET = 0.25 - HEAVE_TARGET_OFFSET
HEAVE_ACCEPTABLE_ERROR = 0.05
HEAVE_OFFSET = 0

PITCH_TARGET_OFFSET = -5
PITCH_KP = 0.4
PITCH_KI = 0.001
PITCH_KD = 0
PITCH_TARGET = 0 - PITCH_TARGET_OFFSET
PITCH_ACCEPTABLE_ERROR = 1
PITCH_OFFSET = 0

ROLL_KP = -0.2
ROLL_KI = 0
ROLL_KD = 0
ROLL_TARGET = 3
ROLL_ACCEPTABLE_ERROR = 1

YAW_KP = 5
YAW_KI = 0
YAW_KD = 0
YAW_TARGET  = 270
YAW_ACCEPTABLE_ERROR = 1




def detection_callback(msg):
    global detections
    detections = list(msg.data) if msg.data else []

# def frame_callback(msg):
#     global frame
#     np_arr = np.frombuffer(msg.data, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
#     if frame is None:
#         return
#     center_x = FRAME_WIDTH // 2
#     target_x = None
    
#     if detections:
#         for i in range(0, len(detections), 5):
#             xmin, ymin, xmax, ymax, confidence = detections[i:i+5]
#             xmin, ymin, xmax, ymax = int(xmin * frame.shape[1]), int(ymin * frame.shape[0]), int(xmax * frame.shape[1]), int(ymax * frame.shape[0])
#             target_x = (xmin + xmax) // 2
            
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(frame, f"Conf: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             cv2.putText(frame, f"Center: {target_x}", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
class YawControlStateMachine(StateMachine):
    waiting = State(initial=True)
    initializing_sensors = State()
    enabling_heave_pid = State()
    waiting_to_yaw = State()
    adjusting_yaw = State()
    finished = State(final=True)

    start_initializing_sensors = waiting.to(initializing_sensors)
    heave_down = initializing_sensors.to(enabling_heave_pid)
    heave_to_wait_yaw = enabling_heave_pid.to(waiting_to_yaw)
    adjust_yaw = waiting_to_yaw.to(adjusting_yaw)
    wait_yaw = adjusting_yaw.to(waiting_to_yaw)
    yaw_adjustment_done = adjusting_yaw.to(finished)
    yaw_waiting_done = waiting_to_yaw.to(finished)

    def __init__(self):
        self.m = MotionController()
        self.m.set_control_mode(DoF.YAW, ControlMode.OPEN_LOOP)
        self.yaw_thread = None
        self.orientation_sub = None
        self.depth_sub = None
        self.current_yaw = None
        self.current_depth = None
        self.running = False
        super(YawControlStateMachine, self).__init__()

    def on_enter_waiting(self):
        print("Waiting to start.")
        self.m.start()
        time.sleep(0.5)
        self.start_initializing_sensors()

    def on_enter_initializing_sensors(self):
        print("Initializing Sensors.")

        # self.orientation_sub = rospy.Subscriber(
        #     f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation
        # )
        self.depth_sub = rospy.Subscriber(f"/{DATA_SOURCE}/depth", Float32, self.on_depth)

        time.sleep(5)
        self.heave_down()


    def on_enter_enabling_heave_pid(self):
        if PID_STATUS["HEAVE"]:
            print("Enabling Heave PID.")

            self.enable_heave_pid()
            self.set_heave_pid_depth(0.25)

            while (self.current_depth is None 
                   or abs(self.current_depth - HEAVE_TARGET) > HEAVE_ACCEPTABLE_ERROR * 3):
                time.sleep(0.1)

        else:
            print("Heave PID not enabled, skipping to 'waiting to yaw'.")
        
        self.heave_to_wait_yaw()

    def enable_heave_pid(self):
        self.m.set_pid_constants(
            DoF.HEAVE, HEAVE_KP, HEAVE_KI, HEAVE_KD, HEAVE_ACCEPTABLE_ERROR, HEAVE_OFFSET
        )
        self.m.set_pid_limits(DoF.HEAVE, -10, 10, -25, 25)
        self.m.set_control_mode(DoF.HEAVE, ControlMode.CLOSED_LOOP)
    
    def set_heave_pid_depth(self, depth=HEAVE_TARGET) -> None:
        print(f"Depth set to {depth} meters")
        self.m.set_target_point(DoF.HEAVE, depth)

    def disable_heave_pid(self):
        self.m.set_control_mode(DoF.HEAVE, ControlMode.OPEN_LOOP)

    def apply_yaw_thrust(self, t :float, thrust:int, yaw_direction:int):
        applied_thrust = yaw_direction * thrust
        print(f"Applying yaw thrust: {applied_thrust}")
        self.m.set_thrust(DoF.YAW, applied_thrust)
        time.sleep(t)
        self.m.set_thrust(DoF.YAW, RESTING_YAW_THRUST)
        print(f"Yaw thrust set to {RESTING_YAW_THRUST}")
        # time.sleep(t/2)
        

    def on_enter_adjusting_yaw(self, t, thrust,error):
        if not self.running:
            self.running = True
            yaw_direction = 1 if error > 0 else -1
            self.yaw_thread = threading.Thread(target=self.apply_yaw_thrust, args=(t,thrust,yaw_direction), daemon=True)
            self.yaw_thread.start()

    def on_exit_adjusting_yaw(self):
        self.running = False
        if self.yaw_thread and self.yaw_thread.is_alive():
            self.yaw_thread.join()
        self.m.set_thrust(DoF.YAW, RESTING_YAW_THRUST)
        print("Yaw adjustment stopped, thrust set to 0")

yaw_state_machine = YawControlStateMachine()

def process_detections():

    centerX = 0.50
    # for i in range(0, len(detections), 5):
    for i in range(0, 5, 5):
        if i + 4 >= len(detections):
            continue
        xmin, ymin, xmax, ymax, confidence = detections[i:i+5]
        targetX = (xmin + xmax) / 2
        targetY = (ymin + ymax) / 2
        aspectRatio = (xmax - xmin) / (ymax - ymin)
        print(f"Aspect Ratio: {aspectRatio}")

        error = targetX - centerX

        if abs(error) > OFFSET and yaw_state_machine.is_waiting_to_yaw:
            YAW_ADJUSTMENT_THRUST = error * YAW_THRUST
            YAW_ADJUSTMENT_TIME = error * YAW_TIME
            yaw_state_machine.adjust_yaw(YAW_ADJUSTMENT_TIME,YAW_ADJUSTMENT_THRUST, error)
        elif abs(error) <= OFFSET and yaw_state_machine.is_adjusting_yaw:
            yaw_state_machine.wait_yaw()

def main():
    rospy.Subscriber('/yolo_detections', Float32MultiArray, detection_callback)
    # rospy.Subscriber('/yolo_frame', CompressedImage, frame_callback)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        process_detections()
        rate.sleep()
    try:
        yaw_state_machine.yaw_adjustment_done()
    except:
        yaw_state_machine.yaw_waiting_done()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

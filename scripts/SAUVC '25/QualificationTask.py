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

FRAME_WIDTH = 640
OFFSET = 40 / FRAME_WIDTH

RESTING_YAW_THRUST = 0
YAW_THRUST = 40
YAW_ADJUSTMENT_THRUST = None
YAW_TIME = 0.3
YAW_ADJUSTMENT_TIME = None
YAW_SEARCH_THRUST = 20

SURGE_THRUST = 40

AREA_THRESHOLD = 0.7 # (0,1)

DATA_SOURCE = "sensors"
FIRST_OR_DATA = True

PID_STATUS = {
    "HEAVE": True,
    "PITCH": False,
    "ROLL": False,
    "YAW": False
}

SAFETY_LIMITS = {
    "DEPTH": 0.1,
    "PITCH": 10,
    "ROLL": 10,
    "YAW": 10
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

class QualificationTask(StateMachine):

    # States
    waiting = State(initial=True)
    initializing_sensors = State()
    enabling_heave_pid = State()
    waiting_to_yaw = State()
    adjusting_yaw = State()
    surging = State()
    safety_routine = State()
    gate_crossed = State()
    surfacing = State()
    finished = State(final=True)

    # Transitions
        # Initalization
    initialize_sensors = waiting.to(initializing_sensors)
    enable_heave_pid_trans = initializing_sensors.to(enabling_heave_pid)
    heave_to_wait = enabling_heave_pid.to(waiting_to_yaw)

        # Central loop
    adjust_yaw = waiting_to_yaw.to(adjusting_yaw)
    wait_yaw = adjusting_yaw.to(waiting_to_yaw)
    yaw_to_surge = adjusting_yaw.to(surging)
    surge_to_yaw = surging.to(adjusting_yaw)

        # Safety routine
    surge_to_safety = surging.to(safety_routine)
    yaw_to_safety = adjusting_yaw.to(safety_routine)
    wait_to_safety = waiting_to_yaw.to(safety_routine)
    safety_to_wait = safety_routine.to(waiting_to_yaw)

        # Gate crossed
    surge_to_gate_crossed = surging.to(gate_crossed)
    gate_crossed_to_surfacing = gate_crossed.to(surfacing)

        # Finishing
    surfacing_to_finished = surfacing.to(finished)
    wait_to_finished = waiting_to_yaw.to(finished)
    adjusting_yaw_to_finished = adjusting_yaw.to(finished)
    surging_to_finished = surging.to(finished)


    def __init__(self):
        self.m = MotionController()
        

        self.initial_orientation = [0, 0, 0] # [roll, pitch, yaw]
        self.current_orientation = [0, 0, 0]
        self.current_depth = None

        self.orientation_sub = None
        self.depth_sub = None
        

        self.yaw_thread = None
        self.surge_thread = None
        self.running = False
        self.surge_running = False
        self.safe_to_cross_gate = False
        super(QualificationTask, self).__init__()

    def on_enter_waiting(self):
        print("Waiting to start.")
        self.m.start()
        time.sleep(0.5)
        self.start_initializing_sensors()

    def on_enter_initializing_sensors(self):
        print("Initializing Sensors.")

        self.orientation_sub = rospy.Subscriber(f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation)
        self.depth_sub = rospy.Subscriber(f"/{DATA_SOURCE}/depth", Float32, self.on_depth)

        self.m.set_control_mode(DoF.YAW, ControlMode.OPEN_LOOP)
        time.sleep(4)
        self.enable_heave_pid_trans()

    def on_enter_enabling_heave_pid(self):
        if PID_STATUS["HEAVE"]:
            print("Enabling Heave PID.")

            self.enable_heave_pid()
            self.set_heave_pid_depth(HEAVE_TARGET)

            while (self.current_depth is None 
                   or abs(self.current_depth - HEAVE_TARGET) > HEAVE_ACCEPTABLE_ERROR):
                time.sleep(0.1)

        else:
            print("Heave PID not enabled, skipping to 'waiting to yaw'.")
        time.sleep(0.2)
        print(f"Saving current orientation as initial orientation.{self.current_orientation}")
        self.initial_orientation = self.current_orientation
        self.heave_to_wait()

    def on_enter_waiting_to_yaw(self):
        print("Waiting to yaw.")  
        

    def on_enter_adjusting_yaw(self, t, thrust,error):
        print("Adjusting yaw.")
        if not self.running:
            self.running = True
            yaw_direction = 1 if error > 0 else -1
            self.yaw_thread = threading.Thread(target=self.apply_yaw, args=(t,thrust,yaw_direction), daemon=True)
            self.yaw_thread.start()

    def on_exit_adjusting_yaw(self):
        self.running = False
        if self.yaw_thread and self.yaw_thread.is_alive():
            self.yaw_thread.join()
        self.m.set_thrust(DoF.YAW, RESTING_YAW_THRUST)
        print("Yaw adjustment stopped, thrust set to 0")


    def on_enter_surging(self, t, thrust):
        print("Surging...")
        if not self.surge_running:
            self.surge_running = True
            self.surge_thread = threading.Thread(target=self.apply_surge, args=(t,thrust), daemon=True)
            self.surge_thread.start()

    def on_exit_surging(self):
        self.surge_running = False
        if self.surge_thread and self.surge_thread.is_alive():
            self.surge_thread.join()
        self.m.set_thrust(DoF.SURGE, 0)
        print("Surge stopped, thrust set to 0")


    def on_enter_safety_routine(self):
        print("Safety compromised, stopping all motion.")
        self.m.setcontrol_mode(DoF.YAW, ControlMode.OPEN_LOOP)
        self.m.setcontrol_mode(DoF.PITCH, ControlMode.OPEN_LOOP)
        self.m.setcontrol_mode(DoF.ROLL, ControlMode.OPEN_LOOP)
        self.m.setthrust(DoF.YAW, 0)
        self.m.setthrust(DoF.PITCH, 0)
        self.m.setthrust(DoF.ROLL, 0)
        self.m.setthrust(DoF.SURGE, 0)
        time.sleep(2)
        print("Safety routine completed. Resuming...")
        self.safety_to_wait()


    def on_enter_gate_crossed(self):
        pass

    def on_enter_surfacing(self):
        print("Surfacing...")
        self.m.setcontrol_mode(DoF.HEAVE, ControlMode.OPEN_LOOP)
        self.m.setthrust(DoF.HEAVE, 0)
        self.surfacing_to_finished()


    def on_orientation(self, vec: Vector3):
        if FIRST_OR_DATA:
            self.initial_orientation = [vec.x, vec.y, vec.z]
            FIRST_OR_DATA = False
        self.current_orientation = [vec.x, vec.y, vec.z]
        self.m.set_current_point(DoF.ROLL, vec.x)
        self.m.set_current_point(DoF.PITCH, vec.y)
        self.m.set_current_point(DoF.YAW, vec.z)
        print(f'Current Orientation : {self.current_orientation}')

    def on_depth(self, depth: Float32):
        self.current_depth = depth.data
        self.m.set_current_point(DoF.HEAVE, depth.data)
        print(f'Current Depth : {self.current_depth}')

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

    def apply_yaw(self, t :float, thrust:int, yaw_direction:int):
        applied_thrust = yaw_direction * thrust
        print(f"Applying yaw thrust: {applied_thrust}")
        self.m.set_thrust(DoF.YAW, applied_thrust)
        time.sleep(t)
        self.m.set_thrust(DoF.YAW, RESTING_YAW_THRUST)
        print(f"Yaw thrust set to {RESTING_YAW_THRUST}")
        # time.sleep(t/2)

    def apply_surge(self):
        print(f"Applying surge thrust: {SURGE_THRUST}")
        self.m.set_thrust(DoF.SURGE, SURGE_THRUST)

        centerX = 0.50
        gate_area_array = []
        while 1:

            if abs(self.initial_orientation[1] - self.current_orientation[1]) > SAFETY_LIMITS["PITCH"]:
                self.surge_to_safety()
            elif abs(self.initial_orientation[0] - self.current_orientation[0]) > SAFETY_LIMITS["ROLL"]:
                self.surge_to_safety()
            elif abs(self.initial_orientation[2] - self.current_orientation[2]) > SAFETY_LIMITS["YAW"]:
                self.surge_to_safety()

            i=0
            if i + 4 >= len(detections):
                continue
            if len(detections) != 0:
                xmin, ymin, xmax, ymax, confidence = detections[i:i+5]
                targetX = (xmin + xmax) / 2
                targetY = (ymin + ymax) / 2
                aspectRatio = (xmax - xmin) / (ymax - ymin)
                gate_area = (xmax - xmin) * (ymax - ymin)
                gate_area_array.append(gate_area)
                if len(gate_area_array) > 5:
                    gate_area_array.pop(0)

                error = targetX - centerX
                print(f'Error : {error}')
                if abs(error) > OFFSET:
                    break
                elif sum(gate_area_array)/5 > AREA_THRESHOLD:
                    self.safe_to_cross_gate = True
                    break
            
            

        self.m.set_thrust(DoF.SURGE, 0)
        print(f"Surge thrust set to 0")
    
    # def perform_safety_checks(self):
    #     if self.initial_orientation[1] > SAFETY_LIMITS["PITCH"]:
    #         self.yaw_to_safety()
    #     elif self.initial_orientation[0] > SAFETY_LIMITS["ROLL"]:
    #         self.yaw_to_safety()
    #     elif self.initial_orientation[2] > SAFETY_LIMITS["YAW"]:
    #         self.yaw_to_safety()

QualificationStateMachine = QualificationTask()


def process_detections():

    if detections == []:
        if QualificationStateMachine.is_surging:
            QualificationStateMachine.surge_to_yaw()
        elif QualificationStateMachine.is_adjusting_yaw:
            QualificationStateMachine.wait_yaw()

    centerX = 0.50
    # for i in range(0, len(detections), 5):
    for i in range(0, 5, 5):
        if i + 4 >= len(detections):
            continue
        xmin, ymin, xmax, ymax, confidence = detections[i:i+5]
        targetX = (xmin + xmax) / 2
        targetY = (ymin + ymax) / 2
        aspectRatio = (xmax - xmin) / (ymax - ymin)
        # print(f"Aspect Ratio: {aspectRatio}")

        error = targetX - centerX
        print(f'Error : {error}')
        if abs(error) > OFFSET:      
            if QualificationStateMachine.is_waiting_to_yaw:
                YAW_ADJUSTMENT_THRUST = error * YAW_THRUST
                YAW_ADJUSTMENT_TIME = error * YAW_TIME
                QualificationStateMachine.adjust_yaw(YAW_ADJUSTMENT_TIME,YAW_ADJUSTMENT_THRUST, error)
            elif QualificationStateMachine.is_surging:
                QualificationStateMachine.surge_to_yaw()

        elif abs(error) <= OFFSET and QualificationStateMachine.is_adjusting_yaw:
            QualificationStateMachine.yaw_to_surge()



def main():
    rospy.Subscriber('/yolo_detections', Float32MultiArray, detection_callback)
    # rospy.Subscriber('/yolo_frame', CompressedImage, frame_callback)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        process_detections()
        rate.sleep()
    try:
        QualificationStateMachine.wait_to_finished()
    except:
        try:
            QualificationStateMachine.surge_to_finished()
        except:
            QualificationStateMachine.adjusting_yaw_to_finished()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
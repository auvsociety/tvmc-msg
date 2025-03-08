#!/usr/bin/env python3
import rospy
import cv2, threading, time
import numpy as np
from statemachine import StateMachine, State
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, Int16, Float32
from geometry_msgs.msg import Vector3
from tvmc import MotionController, DoF, ControlMode
from rose_tvmc_msg.msg import LEDControl


detections = []

FRAME_WIDTH = 640
OFFSET = 40 / FRAME_WIDTH

RESTING_YAW_THRUST = 0
YAW_THRUST = 100
YAW_ADJUSTMENT_THRUST = None
YAW_TIME = 0.35
YAW_ADJUSTMENT_TIME = None
YAW_SEARCH_THRUST = 20

YAW_DEVIATION_THRESHOLD = 5

SURGE_THRUST = 60

AREA_THRESHOLD = 0.65 # (0,1)


DATA_SOURCE = "sensors"

PID_STATUS = {
    "HEAVE": True,
    "PITCH": False,
    "ROLL": False,
    "YAW": False
}

HEAVE_TARGET_OFFSET = -0.07
HEAVE_KP = -45
HEAVE_KI = -0.05
HEAVE_KD =  25
HEAVE_TARGET = 0.33 - HEAVE_TARGET_OFFSET
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

    FIRST_OR_DATA = None

    waiting = State(initial=True, value=0)
    initializing_sensors = State(value=1)
    enabling_heave_pid = State(value=2)
    waiting_to_yaw = State(value=3)
    adjusting_yaw = State(value=4)
    surging = State(value=5)
    safety_routine = State(value=6)
    gate_crossed = State(value=7)
    surfacing = State(value=8)
    finished = State()


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

    
    surge_to_gate = surging.to(gate_crossed)
    gate_to_surface = gate_crossed.to(surfacing)

    wait_to_safety = waiting_to_yaw.to(safety_routine)
    yaw_to_safety = adjusting_yaw.to(safety_routine)
    surge_to_safety = surging.to(safety_routine)
    
    safety_to_adjust = safety_routine.to(adjust_yaw)

    wait_to_finished = waiting_to_yaw.to(finished)
    adjusting_yaw_to_finished = adjusting_yaw.to(finished)
    surging_to_finished = surging.to(finished)
    surface_to_finished = surfacing.to(finished)

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

        self.ledpub = rospy.Publisher('/control/led', Int16,queue_size=5)

        super(QualificationTask, self).__init__()

    def run_safety_checks(self):
        # if abs(self.initial_orientation[0] - self.current_orientation[0]) or abs(self.initial_orientation[1] - self.current_orientation[1]) or abs(self.initial_orientation[2] - self.current_orientation[2]):
        if abs(self.initial_orientation[2] - self.current_orientation[2]) > YAW_DEVIATION_THRESHOLD:
            if self.current_state_value == 4:
                try:
                    self.yaw_to_safety()
                except:
                    print("Safety from yaw failed")
            elif self.current_state_value == 5:
                try:
                    self.surge_to_safety()
                except:
                    print("Safety from surging failed")
            elif self.current_state_value == 3:
                try:
                    self.wait_to_safety()
                except:
                    print("Safety from waiting failed")
    
    def on_enter_safety_routine(self):
        self.ledpub.publish(84)
        print("safety compromised. adjusting")
        self.m.set_thrust(DoF.YAW, 0)
        self.m.set_thrust(DoF.SURGE, 0)
        dev = self.initial_orientation[2] - self.current_orientation[2]
        self.safety_to_adjust(0.05,10,dev/abs(dev))



    def on_enter_waiting(self):
        print("Waiting to start.")
        self.m.start()
        time.sleep(0.5)
        self.initialize_sensors()
    
    def on_enter_initializing_sensors(self):
        print("Initializing Sensors.")

        self.orientation_sub = rospy.Subscriber(f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation)
        self.depth_sub = rospy.Subscriber(f"/{DATA_SOURCE}/depth", Float32, self.on_depth)

        self.m.set_control_mode(DoF.YAW, ControlMode.OPEN_LOOP)
        self.m.set_control_mode(DoF.PITCH, ControlMode.OPEN_LOOP)
        self.m.set_control_mode(DoF.ROLL, ControlMode.OPEN_LOOP)
        self.ledpub.publish(1)
        time.sleep(3)
        self.enable_heave_pid_trans()


    def on_enter_enabling_heave_pid(self):
        self.ledpub.publish(9)
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
        self.ledpub.publish(292)
        print("Waiting to yaw.")

    def on_enter_adjusting_yaw(self, t, thrust,error):
        self.ledpub.publish(146)
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


    def on_enter_surging(self):
        self.ledpub.publish(138)
        print("Surging...")
        if not self.surge_running:
            self.surge_running = True
            self.surge_thread = threading.Thread(target=self.apply_surge, daemon=True)
            self.surge_thread.start()

    def on_exit_surging(self):
        self.surge_running = False
        if self.surge_thread and self.surge_thread.is_alive():
            self.surge_thread.join()
        self.m.set_thrust(DoF.SURGE, 0)
        print("Surge stopped, thrust set to 0")
    
    def on_enter_finished(self):
        self.ledpub.publish(511)
        print("Finished")
    
    def on_enter_gate_crossed(self):
        print("Crossing Gate")
        self.ledpub.publish(78)
        self.m.set_thrust(DoF.SURGE, 60)
        time.sleep(2.5)
        self.m.set_thrust(DoF.SURGE, 0)
        time.sleep(1)
        self.m.set_thrust(DoF.YAW, -60)
        while 1:
            err = self.target_orientation[2] - self.current_orientation[2] + 360 % 360
            if err > 180:
                err -= 360
            if err >175:
                break
            time.sleep(0.1)
        self.m.set_thrust(DoF.YAW, 0)
        time.sleep(1)
        self.m.set_thrust(DoF.SURGE, 60)
        time.sleep(3)
        self.m.set_thrust(DoF.SURGE, 0)
        self.gate_to_surface()

    def on_enter_surfacing(self):
        print("Surfacing..")
        self.ledpub.publish(511)
        self.m.set_control_mode(DoF.HEAVE, ControlMode.OPEN_LOOP)
        self.surface_to_finished()








    def on_orientation(self, vec: Vector3):
        #if FIRST_OR_DATA is not None and FIRST_OR_DATA==True:
        #    self.initial_orientation = [vec.x, vec.y, vec.z]
        #    FIRST_OR_DATA = False
        self.current_orientation = [vec.x, vec.y, vec.z]
        self.m.set_current_point(DoF.ROLL, vec.x)
        self.m.set_current_point(DoF.PITCH, vec.y)
        self.m.set_current_point(DoF.YAW, vec.z)

    def on_depth(self, depth: Float32):
        self.current_depth = depth.data
        self.m.set_current_point(DoF.HEAVE, depth.data)

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

    def apply_surge(self):
        tn = time.time()
        print(f"Applying surge thrust: {SURGE_THRUST}")
        self.m.set_thrust(DoF.SURGE, SURGE_THRUST)

        centerX = 0.50
        gate_area_array = []
        while self.current_state_value == 5:
            i=0
            if i + 4 >= len(detections):
                continue
            xmin, ymin, xmax, ymax, confidence = detections[i:i+5]
            targetX = (xmin + xmax) / 2
            targetY = (ymin + ymax) / 2
            aspectRatio = (xmax - xmin) / (ymax - ymin)
            gate_area = (xmax - xmin) * (ymax - ymin)
            gate_area_array.append(xmax-xmin)
            if len(gate_area_array) > 2:
                gate_area_array.pop(0)

            error = targetX - centerX
            # print(f'Error : {error}')
            if abs(error) > OFFSET:
                break
            elif sum(gate_area_array)/2 > AREA_THRESHOLD:
                self.safe_to_cross_gate = True
                self.surge_to_gate()
                break
            
            

        self.m.set_thrust(DoF.SURGE, 0)
        print(f"Surge thrust set to 0")
        self.surge_to_yaw(0,0,0)


QualificationStateMachine = QualificationTask()



def process_detections():

    if detections == []:
         if QualificationStateMachine.current_state == 5:
             QualificationStateMachine.surge_to_yaw()
    #     elif QualificationStateMachine.current_state_value == 4:
    #         QualificationStateMachine.wait_yaw()

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
    
        if abs(error) > OFFSET:
            YAW_ADJUSTMENT_THRUST = abs(error) * YAW_THRUST
            YAW_ADJUSTMENT_TIME = abs(error) * YAW_TIME
            if QualificationStateMachine.current_state_value == 3:
                QualificationStateMachine.adjust_yaw(YAW_ADJUSTMENT_TIME,YAW_ADJUSTMENT_THRUST, error)
            elif QualificationStateMachine.current_state_value == 4:
                QualificationStateMachine.wait_yaw()
            elif QualificationStateMachine.current_state_value == 5:
               try:
                   QualificationStateMachine.surge_to_yaw(YAW_ADJUSTMENT_TIME,YAW_ADJUSTMENT_THRUST, error)
               except:
                   print("Exception")
                            
        elif abs(error) <= OFFSET:
            if QualificationStateMachine.current_state_value == 4:
                QualificationStateMachine.yaw_to_surge()
            if QualificationStateMachine.current_state_value == 3:
                QualificationStateMachine.adjust_yaw(0,0,0)
                QualificationStateMachine.yaw_to_surge()


def main():
    rospy.Subscriber('/yolo_detections', Float32MultiArray, detection_callback)
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

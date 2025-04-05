#!/usr/bin/env python3
import rospy, cv2, threading, time
import numpy as np
from statemachine import StateMachine, State
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, Int16, Float32
from geometry_msgs.msg import Vector3
from tvmc import MotionController, DoF, ControlMode

from PID_CONSTANTS import *
# ---------------------------------------------------------------------------------------------------
DATA_SOURCE = "sensors"

HEAVE_TARGET_OFFSET = -0.08
HEAVE_KP = -25 # -90 #-70 #-60 #-40 #-50 # -100
HEAVE_KI = 0
HEAVE_KD = 60 #30# 5.2 #6.5
HEAVE_TARGET = 1.45 - HEAVE_TARGET_OFFSET
HEAVE_ACCEPTABLE_ERROR = 0.05
HEAVE_OFFSET = 0 #-0.13 # 0

PITCH_TARGET_OFFSET = -5
PITCH_KP = -0.25#-0.25  #0.8
PITCH_KI = 0#0.02
PITCH_KD = 50# 0.15 #0.2
PITCH_TARGET = 0 - PITCH_TARGET_OFFSET
PITCH_ACCEPTABLE_ERROR = 1
PITCH_OFFSET = 0 #5

ROLL_KP = 0#-0.2 #0.1
ROLL_KI = 0
ROLL_KD = 0
ROLL_TARGET = 3
ROLL_ACCEPTABLE_ERROR = 1

YAW_TARGET_OFFSET = -5
YAW_KP = -2
YAW_KI = 0
YAW_KD = 10
YAW_TARGET  = 161 - YAW_TARGET_OFFSET
YAW_ACCEPTABLE_ERROR = 0.8
# ---------------------------------------------------------------------------------------------------





detections = []
SURGE_BOOL = False
ACTIVITY_STAT = True
PID_STATUS = {
    "HEAVE" :True,
    "PITCH" :True,
    "YAW" :True
}
'''
PID_STATUS = {
    "HEAVE" :False,
    "PITCH" :False,
    "YAW" :False
}
'''
# Time durations
# ---------------------------------------------------------------------------------------------------
PRE_SURGE_TIME = 3
NAV_SURGE_TIME = 5
# ---------------------------------------------------------------------------------------------------



# 
# ---------------------------------------------------------------------------------------------------
FRAME_WIDTH = 640
OFFSET = 40 / FRAME_WIDTH
# ---------------------------------------------------------------------------------------------------



# Thrust values
# ---------------------------------------------------------------------------------------------------
RESTING_YAW_THRUST = 0
YAW_THRUST = 60
YAW_ADJUSTMENT_THRUST = None
# YAW_TIME = 0.25
# YAW_ADJUSTMENT_TIME = None
YAW_SEARCH_THRUST = 20
SURGE_THRUST = 20
# ---------------------------------------------------------------------------------------------------


def detection_callback(msg):
    global detections
    detections = list(msg.data) if msg.data else []


def frame_callback(msg):
    global frame
    np_arr = np.frombuffer(msg.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


class FullDrive(StateMachine):

    # States
    # ---------------------------------------------------------------------------------------------------
    waiting = State(initial=True, value=0)
    initializing_sensors = State(value=1)
    enabling_pids = State(value=2)
    pre_surge = State(value=3)
    nav_search = State(value=4)
    nav_yaw_lock = State(value=5)
    nav_approach = State(value=6)
    blind_surge = State()

    finished = State(final=True)
    # ---------------------------------------------------------------------------------------------------




    # Transitions
    # ---------------------------------------------------------------------------------------------------
    trans_to_initializing_sensors = waiting.to(initializing_sensors)
    trans_to_enabling_pids = initializing_sensors.to(enabling_pids)
    trans_to_pre_surge = enabling_pids.to(pre_surge)
    trans_to_nav_search = pre_surge.to(nav_search)
    trans_to_nav_yaw_lock = nav_search.to(nav_yaw_lock)
    trans_to_nav_approach = nav_yaw_lock.to(nav_approach)
    trans_to_blind_surge = nav_approach.to(blind_surge)
    trans_to_finished = blind_surge.to(finished)

    # ---------------------------------------------------------------------------------------------------


    def __init__(self):
        self.m = MotionController()
        

        self.initial_orientation = [0, 0, 0] # [roll, pitch, yaw]
        self.current_orientation = [0, 0, 0]
        self.current_depth = None
        self.yaw_lock = None
        self.current_yaw = None
        
        self.gate_yaw = None

        self.orientation_sub = None
        self.depth_sub = None
        
        self.gate_crossed = False

        self.surge_thread = None

        self.ledpub = rospy.Publisher('/control/led', Int16,queue_size=5)

        super(FullDrive, self).__init__()        


    def on_enter_waiting(self):
        print("Waiting...")
        self.m.start()
        time.sleep(0.5)
        self.trans_to_initializing_sensors()

    
    def on_enter_initializing_sensors(self):
        print("Initializing Sensors...")

        self.orientation_sub = rospy.Subscriber(f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation)
        self.depth_sub = rospy.Subscriber(f"/{DATA_SOURCE}/depth", Float32, self.on_depth)

        self.m.set_control_mode(DoF.ROLL, ControlMode.OPEN_LOOP)
        self.ledpub.publish(1)
        time.sleep(3)

        print(f"Saving current orientation as initial orientation: {self.current_orientation}")
        self.initial_orientation = self.current_orientation


        # yaw locking using samples
        samples = []

        while self.current_yaw is None:
            time.sleep(0.1)
        self.yaw_lock = self.current_yaw
        current_time = time.time()

        while time.time() - current_time < 1:
            samples.append(self.current_yaw)
            time.sleep(0.1)
        samples = [(x - 360 if x >= 180 else x) for x in samples]
        print(samples)
        self.yaw_lock = (sum(samples) / len(samples))
        self.yaw_lock -= 5
        self.yaw_lock = self.yaw_lock + 360 if self.yaw_lock < 0 else self.yaw_lock
        print("Yaw locked at: ", self.yaw_lock)

        self.trans_to_enabling_pids()
    
    def on_enter_enabling_pids(self):
        self.ledpub.publish(9)
        if PID_STATUS["HEAVE"]:
            print("Enabling Heave PID.")
            self.m.set_pid_constants(
                DoF.HEAVE, HEAVE_KP, HEAVE_KI, HEAVE_KD, HEAVE_ACCEPTABLE_ERROR, HEAVE_OFFSET
            )
            self.m.set_pid_limits(DoF.HEAVE, -10, 10, -25, 25)
            self.m.set_target_point(DoF.HEAVE, HEAVE_TARGET)
            self.m.set_control_mode(DoF.HEAVE, ControlMode.CLOSED_LOOP)

            while (self.current_depth is None 
                    or abs(self.current_depth - HEAVE_TARGET + HEAVE_TARGET_OFFSET) > HEAVE_ACCEPTABLE_ERROR * 3):
                time.sleep(0.1)

            time.sleep(1.5)


        if PID_STATUS["PITCH"]:
            print("Enabling Pitch PID.")
            self.m.set_pid_constants(
                DoF.PITCH, PITCH_KP, PITCH_KI, PITCH_KD, PITCH_ACCEPTABLE_ERROR, PITCH_OFFSET
            )
            self.m.set_pid_limits(DoF.PITCH, -10, 10, -25, 25)
            self.m.set_target_point(DoF.PITCH, PITCH_TARGET)
            self.m.set_control_mode(DoF.PITCH, ControlMode.CLOSED_LOOP)
            time.sleep(0.5)


        if PID_STATUS["YAW"]:
            print("Enabling Yaw PID.")

            self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
            self.m.set_pid_limits(DoF.YAW, -10, 10, -25, 25)
            self.m.set_target_point(DoF.YAW, self.yaw_lock)
            self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)


            while (self.current_yaw is None or abs(self.current_yaw - self.yaw_lock) > YAW_ACCEPTABLE_ERROR):
                print("Current yaw: ", self.current_yaw)
                time.sleep(0.1)

            # while (self.current_yaw is None 
            #         or abs(self.current_yaw - YAW_TARGET + YAW_TARGET_OFFSET) > YAW_ACCEPTABLE_ERROR):
            #     time.sleep(0.1)
            
            time.sleep(1.5)

        self.trans_to_pre_surge()


    def on_enter_pre_surge(self):
        print("Pre surging")
        self.m.set_thrust(DoF.SURGE, 50)
        time.sleep(PRE_SURGE_TIME)
        self.m.set_thrust(DoF.SURGE, 0)
        self.trans_to_nav_search()

    def on_enter_nav_search(self):
        flag = False
        for i in range(0,60,5):
            print("YAW LOCKED", self.yaw_lock +i)
            self.m.set_target_point(DoF.YAW, (self.yaw_lock - i))
            time.sleep(3)
                
            centerX = 0.5
            for j in range(0, len(detections), 5):
                if j + 4 >= len(detections):
                    continue
                xmin, ymin, xmax, ymax, confidence = detections[j:j+5]
                targetX = (xmin + xmax) / 2
                targetY = (ymin + ymax) / 2
                aspectRatio = (xmax - xmin) / (ymax - ymin)
                print(f"Aspect Ratio: {aspectRatio}")
                if aspectRatio >0.2 and aspectRatio <0.9:
                    error = targetX - centerX
                    print(f'detected :{aspectRatio}')

                    if abs(error) < OFFSET:
                        self.gate_yaw = self.current_yaw
                        flag = True
                        break
                    else:
                        self.gate_yaw = self.current_yaw - error*10
                        flag = True
                        break

            if flag == True:
                break
        if flag != True:
            self.m.set_target_point(DoF.YAW, self.yaw_lock)
            
            print("YAW LOCKED", self.yaw_lock)
            time.sleep(4)
        
            for i in range(0,60,5):
                print("YAW LOCKED", self.yaw_lock +i)
                self.m.set_target_point(DoF.YAW, (self.yaw_lock + i))
                time.sleep(3)
                    
                centerX = 0.5
                for j in range(0, len(detections), 5):
                    if j + 4 >= len(detections):
                        continue
                    xmin, ymin, xmax, ymax, confidence = detections[j:j+5]
                    targetX = (xmin + xmax) / 2
                    targetY = (ymin + ymax) / 2
                    aspectRatio = (xmax - xmin) / (ymax - ymin)
                    print(f"Aspect Ratio: {aspectRatio}")
                    if aspectRatio >0.2 and aspectRatio <0.8:
                        error = targetX - centerX
                        print(f'detected :{aspectRatio}')

                        if abs(error) < OFFSET:
                            self.gate_yaw = self.current_yaw
                            flag = True
                            break
                        else:
                            self.gate_yaw = self.current_yaw - error *10
                            flag = True
                            break
                if flag == True:
                    break
        if self.gate_yaw == None:
            print("Gate Not found")

        self.trans_to_nav_yaw_lock()
        
    def on_enter_nav_yaw_lock(self):
        self.ledpub.publish(18)
        self.yaw_lock = self.gate_yaw
        print("Attempting to lock nav yaw. ", self.yaw_lock)
        self.m.set_target_point(DoF.YAW, self.yaw_lock)

        while (self.current_yaw is None or abs(self.current_yaw - self.yaw_lock) > YAW_ACCEPTABLE_ERROR):
            print("Current yaw: ", self.current_yaw)
            time.sleep(0.1)
        
        time.sleep(1.5)

        self.trans_to_nav_approach()

    
    def on_enter_nav_approach(self):
        global SURGE_BOOL, ACTIVITY_STAT
        self.ledpub.publish(511)
        self.surge_thread = threading.Thread(target=self.surge_thread_func, daemon=True)
        self.surge_thread.start()

        c_time = time.time()
        while self.gate_crossed == False:
            centerX = 0.5
            for j in range(0, len(detections), 5):
                if j + 4 >= len(detections):
                    continue
                xmin, ymin, xmax, ymax, confidence = detections[j:j+5]
                targetX = (xmin + xmax) / 2
                targetY = (ymin + ymax) / 2
                aspectRatio = (xmax - xmin) / (ymax - ymin)
                print(f"Aspect Ratio: {aspectRatio}")
                if aspectRatio >0.3 and aspectRatio <0.8:
                    
                        
                    error = targetX - centerX
                    if error > 0:
                        print("yaw pid -- 1")

                        self.m.set_target_point(DoF.YAW, self.current_yaw - 1)
                        time.sleep(0.5)
                    elif error < 0:
                        if (xmax - xmin) * 640 > 180:
                            print("Gate crossed")
                            self.gate_crossed = True
                            self.m.set_target_point(DoF.YAW, self.current_yaw)
                            self.trans_to_blind_surge()
                            time.sleep(0.5)
                        print("yaw pid ++ 1")
                        self.m.set_target_point(DoF.YAW, self.current_yaw + 1)
                        time.sleep(0.5)    
                    if abs(error) < OFFSET:
                        print("Surge bool to True... ")
                        SURGE_BOOL = True
                    else:
                        print("Surge bool to False... ")
                        SURGE_BOOL = False     
                    if (xmax - xmin) * 640 > 180:
                        print("Gate crossed")
                        self.gate_crossed = True
                        self.m.set_target_point(DoF.YAW, self.current_yaw)
                        #self.trans_to_blind_surge()
                        time.sleep(0.5)
        
        SURGE_BOOL = False
        ACTIVITY_STAT = False

    def on_enter_blind_surge(self):
        print("Blind SURGE started")
        self.ledpub.publish(127)
        self.m.set_thrust(DoF.SURGE, SURGE_THRUST)
        time.sleep(8)
        self.m.set_thrust(DoF.SURGE, 0)
        self.trans_to_finished()

    def surge_thread_func(self):
        global SURGE_BOOL, ACTIVITY_STAT
        while ACTIVITY_STAT:
            if SURGE_BOOL:
                print("Surge given")
                self.m.set_thrust(DoF.SURGE, SURGE_THRUST)
            else:
                self.m.set_thrust(DoF.SURGE, 0)

    def on_enter_finished(self):
        self.ledpub.publish(63)
        self.m.set_control_mode(DoF.YAW, ControlMode.OPEN_LOOP)
        self.m.set_control_mode(DoF.HEAVE, ControlMode.OPEN_LOOP)
        self.m.set_control_mode(DoF.PITCH, ControlMode.OPEN_LOOP)

        print("Finished")



    # ---------------------------------------------------------------------------------------------------
    def on_orientation(self, vec: Vector3):
        self.current_orientation = [vec.x, vec.y, vec.z]
        self.current_yaw = vec.z
        self.m.set_current_point(DoF.ROLL, vec.x)
        self.m.set_current_point(DoF.PITCH, vec.y)
        self.m.set_current_point(DoF.YAW, vec.z)

    def on_depth(self, depth: Float32):
        self.current_depth = depth.data
        self.m.set_current_point(DoF.HEAVE, depth.data)


        

def main():
    rospy.Subscriber('/yolo_detections', Float32MultiArray, detection_callback)
    rospy.Subscriber('/yolo_frame', CompressedImage, frame_callback)
    TASK = FullDrive()
    rospy.spin()

if __name__ == "__main__":
    main()
    





















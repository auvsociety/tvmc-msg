#!/usr/bin/env python3
import rospy, cv2, threading, time
import numpy as np
from statemachine import StateMachine, State
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, Int16, Float32
from geometry_msgs.msg import Vector3
from tvmc import MotionController, DoF, ControlMode

from PID_CONSTANTS import *

detections = []
frame = None
SURGE_BOOL = False
SURGE_BOOL_ACTIVE = True
SURGE_THRUST = 70

T4_VICINITY_YAW = 30
T4_VICINITY_SURGE_DURATION = 5

PID_STATUS = {
    "HEAVE" :True,
    "PITCH" :True,
    "YAW" :True
}

FRAME_WIDTH = 640
OFFSET = 32 / FRAME_WIDTH

ZERO_YAW = 0
FRONT_YAW = 90


def detection_callback(msg):
    global detections
    detections = list(msg.data) if msg.data else []


def frame_callback(msg):
    global frame
    np_arr = np.frombuffer(msg.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

class CrawlBot(StateMachine):
    waiting = State(initial=True)
    initiliaze = State()
    enable_pids = State()
    yaw_lock_0 = State(value=0)
    surge_state = State(value = 1)
    yaw_lock_90 = State(value=2)
    nav_approach = State()
    blind_surge = State()
    t4_waiting = State()
    t4_locate = State()
    t4_flare_search = State(value=11)
    finished = State()

    to_initialize = waiting.to(initiliaze)
    to_enable_pids = initiliaze.to(enable_pids)

    to_yaw_lock_0 = enable_pids.to(yaw_lock_0)

    to_surge_state_from_yaw_lock_0 = yaw_lock_0.to(surge_state)
    to_yaw_lock_90_from_surge_state = surge_state.to(yaw_lock_90)
    to_surge_state_from_yaw_lock_90 = yaw_lock_90.to(surge_state)
    to_yaw_lock_0_from_yaw_lock_90 = yaw_lock_90.to(yaw_lock_0)
    to_yaw_lock_0_from_surge_state = surge_state.to(yaw_lock_0)


    to_nav_approach = yaw_lock_90.to(nav_approach)
    to_blind_surge = nav_approach.to(blind_surge)

    to_t4_waiting = blind_surge.to(t4_waiting)
    to_t4_locate = t4_waiting.to(t4_locate)
    to_t4_flare_search = t4_locate.to(t4_flare_search)
    to_finished = t4_flare_search.to(finished)

    def __init__(self):
        self.m = MotionController()
        

        self.initial_yaw = None
        self.current_depth = None
        self.yaw_lock = None
        self.current_yaw = None

        self.orientation_sub = None
        self.depth_sub = None
        
        self.gate_crossed = False

        self.surge_thread = None

        self.ledpub = rospy.Publisher('/control/led', Int16,queue_size=5)

        super(CrawlBot, self).__init__() 
    
    def on_enter_waiting(self):
        print("Waiting")
        self.m.start()
        time.sleep(0.5)
        self.to_initialize()

    def on_enter_initiliaze(self):
        print("Initializing Sensors...")
        self.orientation_sub = rospy.Subscriber(f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation)
        self.depth_sub = rospy.Subscriber(f"/{DATA_SOURCE}/depth", Float32, self.on_depth)
        
        self.surge_thread = threading.Thread(target=self.surge)
        self.surge_thread.start()
        
        self.m.set_control_mode(DoF.ROLL, ControlMode.OPEN_LOOP)
        self.ledpub.publish(1)
        time.sleep(3)

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
        self.yaw_lock = self.yaw_lock + 360 if self.yaw_lock < 0 else self.yaw_lock
        print("Yaw locked at: ", self.yaw_lock)
        self.initial_yaw = self.yaw_lock
        self.to_enable_pids()

    def on_enter_enable_pids(self):
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
        
        self.to_yaw_lock_0()
    
    def on_enter_yaw_lock_0(self):
        print("Yaw lock 0")
        self.ledpub.publish(18)
        self.m.set_target_point(DoF.YAW, ZERO_YAW)
        time.sleep(3)

        surge_time = 5

        self.to_surge_state_from_yaw_lock_0(surge_time)
    
    def on_enter_surge_state(self, timeout=5):
        global SURGE_BOOL, SURGE_BOOL_ACTIVE
        SURGE_BOOL_ACTIVE = True
        self.ledpub.publish(36)
        print("Surge state")
        SURGE_BOOL = True
        time.sleep(timeout)
        SURGE_BOOL = False
    
        self.to_yaw_lock_90()
    
    def on_enter_yaw_lock_90(self):
        self.ledpub.publish(9)
        print("Yaw lock 90")
        self.m.set_target_point(DoF.YAW, FRONT_YAW)
        time.sleep(3)
  
    def on_enter_nav_approach(self):
        global SURGE_BOOL
        self.ledpub.publish(511)
        self.surge_thread = threading.Thread(target=self.surge_thread_func, daemon=True)
        self.surge_thread.start()

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
                        print("yaw pid ++ 1")
                        self.m.set_target_point(DoF.YAW, self.current_yaw + 1)
                        time.sleep(0.5)    
                    if abs(error) < OFFSET:
                        print("Surge bool to True... ")
                        SURGE_BOOL = True
                    else:
                        print("Surge bool to False... ")
                        SURGE_BOOL = False     
                    if (xmax - xmin) * 640 > 280:
                        print("Gate crossed")
                        self.m.set_target_point(DoF.YAW, self.current_yaw)
                        time.sleep(0.5)
                        self.to_blind_surge()
        
        SURGE_BOOL = False

    def on_enter_blind_surge(self):
        print("Blind SURGE started")
        self.ledpub.publish(127)
        global SURGE_BOOL_ACTIVE
        SURGE_BOOL_ACTIVE = False
        self.m.set_thrust(DoF.SURGE, SURGE_THRUST)
        time.sleep(5)
        self.m.set_thrust(DoF.SURGE, 0)
        self.to_t4_waiting()

    
    def on_enter_t4_waiting(self):
        print("t4 waiting state")
        time.sleep(1)
        self.to_t4_locate()
    
    def on_enter_t4_locate(self):
        self.ledpub.publish(73)
        print("Moving to vicinity...")
        self.m.set_target_point(DoF.YAW, T4_VICINITY_YAW)
        time.sleep(3)
        global SURGE_BOOL_ACTIVE
        SURGE_BOOL_ACTIVE = False
        print("Moving to site..")
        self.m.set_thrust(DoF.SURGE, SURGE_THRUST)
        time.sleep(T4_VICINITY_SURGE_DURATION)
        self.m.set_thrust(DoF.SURGE, 0)
        self.t4_flare_search()
    
    def on_enter_t4_flare_search(self):
        print("Entering flare search")
        self.ledpub.publish(290)
        self.m.set_target_point(DoF.YAW, 10)
        time.sleep(2)
        c_time = time.time()
        while abs(c_time - time.time()) < 10:
            data = self.detect_flares(frame)
            if len(data) > 0:
                for i in data:
                    error = i[0] - 320
                    if error < 100:
                        print(f"Surging towards {i[2]}")
                        self.m.set_thrust(DoF.SURGE, SURGE_THRUST)
                        time.sleep(3)
                        self.m.set_thrust(DoF.SURGE, 0)
            next_yaw = self.current_yaw + 5
            print(f"Turning to {next_yaw}")
            self.m.set_target_point(DoF.YAW, next_yaw)
            time.sleep(1.5)

        self.to_finished()

    def on_enter_finished(self):
        self.m.set_control_mode(DoF.HEAVE, ControlMode.OPEN_LOOP)
        self.m.set_control_mode(DoF.PITCH, ControlMode.OPEN_LOOP)
        self.m.set_control_mode(DoF.YAW, ControlMode.OPEN_LOOP)
        print("Finished.")
        
    
    def detect_flares(fr):
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0,90,90])
        upper_red = np.array([15,255,255])
        lower_yellow = np.array([15,100,100])
        upper_yellow = np.array([50,255,255])
        lower_blue = np.array([90,60,60])
        upper_blue = np.array([130,255,255])
        
        masks = {
            "red": cv2.inRange(hsv, lower_red, upper_red),
            #"blue": cv2.inRange(hsv, lower_blue, upper_blue),
            "yellow": cv2.inRange(hsv, lower_yellow, upper_yellow)
        }
        
        det = []
        
        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cx,cy = x+w //2, y+h//2
                    det.append([cx,cy,color])
                    
        return det
    
    def surge(self):
        global SURGE_BOOL,SURGE_BOOL_ACTIVE
        if SURGE_BOOL_ACTIVE:
                
            while 1:
                if SURGE_BOOL:
                    print("Surge given")
                    self.m.set_thrust(DoF.SURGE, SURGE_THRUST)
                else:
                    self.m.set_thrust(DoF.SURGE, 0)

    def on_orientation(self, vec: Vector3):
        self.current_yaw = vec.z
        self.m.set_current_point(DoF.PITCH, vec.y)
        self.m.set_current_point(DoF.YAW, vec.z)

    def on_depth(self, depth: Float32):
        self.current_depth = depth.data
        self.m.set_current_point(DoF.HEAVE, depth.data)
    
BotMachine = CrawlBot()

def process_detections():
    if BotMachine.current_state_value == 2:
        centerX = 0.50
        flag = False
        for i in range(0, len(detections), 5):
            if i + 4 >= len(detections):
                continue
            xmin, ymin, xmax, ymax, confidence = detections[i:i+5]
            targetX = (xmin + xmax) / 2
            targetY = (ymin + ymax) / 2
            aspectRatio = (xmax - xmin) / (ymax - ymin)
            print(f"Aspect Ratio: {aspectRatio}")
            if aspectRatio >0.2 and aspectRatio <0.7:
                flag = True
                error = targetX - centerX
                if error > 0:
                    dist_approx = 8.5 * 2 * error
                    BotMachine.m.set_target_point(DoF.YAW, ZERO_YAW)
                elif error < 0:
                    dist_approx = 8.5 * 2 * error
                    BotMachine.m.set_target_point(DoF.YAW, ZERO_YAW + 180)
                SURGE_BOOL_ACTIVE = False
                time.sleep(3)
                BotMachine.m.set_thrust(DoF.SURGE, 1.2*SURGE_THRUST)
                time.sleep(dist_approx)
                BotMachine.m.set_thrust(DoF.SURGE, 0)
                time.sleep(1)
                BotMachine.m.set_target_point(DoF.YAW, FRONT_YAW)
                time.sleep(3)

                BotMachine.to_nav_approach()
        if flag != True:
            BotMachine.to_yaw_lock_0_from_yaw_lock_90()
    


                


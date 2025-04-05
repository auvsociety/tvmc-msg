#!/usr/bin/env python3
import time
import threading

import rospy
from std_msgs.msg import Float32, Int16
from geometry_msgs.msg import Vector3
from statemachine import StateMachine, State
from tvmc import MotionController, DoF, ControlMode


#-----------------------------------------------
DATA_SOURCE = "sensors"

HEAVE_TARGET_OFFSET = -0.08
HEAVE_KP = -25 # -90 #-70 #-60 #-40 #-50 # -100
HEAVE_KI = 0
HEAVE_KD = 60 #30# 5.2 #6.5
HEAVE_TARGET = 0.47 - HEAVE_TARGET_OFFSET
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
#-----------------------------------------------


#from PID_CONSTANTS import *
# pool constants
#-----------------------------------------------
# DATA_SOURCE = "sensors"

# HEAVE_TARGET_OFFSET = -0.08
# HEAVE_KP = -25 # -90 #-70 #-60 #-40 #-50 # -100
# HEAVE_KI = 0
# HEAVE_KD = 60 #30# 5.2 #6.5
# HEAVE_TARGET = 0.35 - HEAVE_TARGET_OFFSET
# HEAVE_ACCEPTABLE_ERROR = 0.05
# HEAVE_OFFSET = 0 #-0.13 # 0

# PITCH_TARGET_OFFSET = -5
# PITCH_KP = -0.25#-0.25  #0.8
# PITCH_KI = 0#0.02
# PITCH_KD = 50# 0.15 #0.2
# PITCH_TARGET = 0 - PITCH_TARGET_OFFSET
# PITCH_ACCEPTABLE_ERROR = 1
# PITCH_OFFSET = 0 #5

# ROLL_KP = 0#-0.2 #0.1
# ROLL_KI = 0
# ROLL_KD = 0
# ROLL_TARGET = 0
# ROLL_ACCEPTABLE_ERROR = 0

# YAW_TARGET_OFFSET = -8#-7.3
# YAW_KP = -2
# YAW_KI = 0
# YAW_KD = 10
# YAW_TARGET  = 161 - YAW_TARGET_OFFSET
# YAW_ACCEPTABLE_ERROR = 0.5
#-----------------------------------------------

YAW_THRUST = 60



#SURGE_THRUST = 70
#ASYNC_CLOCK_DURATION = 13.5

#SURGE_THRUST = 50
#ASYNC_CLOCK_DURATION = 16

SURGE_THRUST = 90
ASYNC_CLOCK_DURAION = 11

ROTATION_YAW = 180


ASYNC2_CLOCK_DURATION = 4
ASYNC_YAW_CLOCK_DURATION = 2.45

class QualificationTask(StateMachine):
    wait_to_start = State(initial=True)
    initializing_sensors = State()
    enabling_heave_pid = State()
    fixing_yaw = State()
    surging_forward = State()
    yaw_180 = State()
    surging_forward_2 = State()
    finished = State(final=True)


    start_initializing_sensors = wait_to_start.to(initializing_sensors)
    heave_down = initializing_sensors.to(enabling_heave_pid)
    fix_yaw = enabling_heave_pid.to(fixing_yaw)
    surge_forward = fixing_yaw.to(surging_forward)
    yaw_180_after_surging = surging_forward.to(yaw_180)
    surging_after_yaw_180 = yaw_180.to(surging_forward_2)
    finish = surging_forward_2.to(finished)


    def __init__(self):
        self.m = MotionController()
        self.yaw_lock = None
        self.orientation_sub = None
        self.depth_sub = None
        self.current_yaw = None
        self.current_depth = None
        self.timer = None
        self.timer_2 = None
        self.led_publisher = rospy.Publisher("/control/led", Int16, queue_size=1)
        self.timer_rot = None
        super(QualificationTask, self).__init__()

    def on_enter_wait_to_start(self):
        self.led_publisher.publish(1)
        print("Waiting to start.")
        self.m.start()
        time.sleep(0)
        self.start_initializing_sensors()

    def on_depth(self, depth: Float32):

        self.current_depth = depth.data
        
        self.m.set_current_point(DoF.HEAVE, depth.data)

    def on_orientation(self, vec: Vector3):
        self.current_yaw = vec.z
        self.m.set_current_point(DoF.YAW, vec.z)


    def on_enter_initializing_sensors(self):
        self.led_publisher.publish(9)
        print("Initializing Sensors.")

        self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
        self.orientation_sub = rospy.Subscriber(
            f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation
        )
        self.depth_sub = rospy.Subscriber(f"/{DATA_SOURCE}/depth", Float32, self.on_depth)

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
        #self.yaw_lock -= (YAW_TARGET_OFFSET + 3.3) # 3.3 to make it a little right for the left tilted surge
        self.yaw_lock -= 5
        self.yaw_lock = self.yaw_lock + 360 if self.yaw_lock < 0 else self.yaw_lock
        print("Yaw locked at: ", self.yaw_lock)
	
        self.heave_down()
    
    def on_enter_enabling_heave_pid(self):
        self.led_publisher.publish(2)
        print("Enabling heave PID.")
        
        self.m.set_pid_constants(DoF.HEAVE, HEAVE_KP, HEAVE_KI, HEAVE_KD, HEAVE_ACCEPTABLE_ERROR, HEAVE_OFFSET)
        self.m.set_pid_limits(DoF.HEAVE, -10, 10, -25, 25)
        self.m.set_control_mode(DoF.HEAVE, ControlMode.CLOSED_LOOP)
        self.m.set_target_point(DoF.HEAVE, HEAVE_TARGET)

        while (self.current_depth is None or abs(self.current_depth - HEAVE_TARGET) > HEAVE_ACCEPTABLE_ERROR * 3):
            time.sleep(0.1)

        self.enable_pitch_pid()

        self.fix_yaw()
    
    def on_enter_fixing_yaw(self):
        self.led_publisher.publish(18)
        print("Attempting to fix yaw. ", self.yaw_lock)
        self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
        self.m.set_target_point(DoF.YAW, self.yaw_lock)
        self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)

        while (self.current_yaw is None or abs(self.current_yaw - self.yaw_lock) > YAW_ACCEPTABLE_ERROR):
            print("Current yaw: ", self.current_yaw)
            time.sleep(0.1)
        
        time.sleep(1.5)

        self.surge_forward()
    
    def on_enter_surging_forward(self):
        self.led_publisher.publish(144)
        print("Surging forward")
        self.m.set_thrust(DoF.SURGE, SURGE_THRUST)

        if self.timer is None:
            self.timer = threading.Thread(target=self.timer_async, daemon=True)
            self.timer.start()

    def on_exit_surging_forward(self):
        print("Stopping Surge.")
        self.m.set_control_mode(DoF.YAW, ControlMode.OPEN_LOOP)
        self.m.set_thrust(DoF.SURGE, 0)

    # manual yaw
    def on_enter_yaw_180(self):
        self.led_publisher.publish(8)
        time.sleep(0.5)
        print("Attempting to yaw 180. manually")
        self.m.set_thrust(DoF.YAW, YAW_THRUST)

        if self.timer_rot is None:
            self.timer_rot = threading.Thread(target=self.timer_async_yaw_manual, daemon=True)
            self.timer_rot.start()


    # # pid yaw
    # def on_enter_yaw_180(self):
    #     self.led_publisher.publish(4)
    #     print("Attempting to yaw 180. using pid")
    #     self.yaw_lock = (self.yaw_lock + ROTATION_YAW) % 360
    #     self.m.set_target_point(DoF.YAW, self.yaw_lock)
    #     print(f"Yaw PID set to {self.yaw_lock}")

    #     while (self.current_yaw is None or abs(self.current_yaw - self.yaw_lock)> YAW_ACCEPTABLE_ERROR):
    #         time.sleep(0.5)
    #     time.sleep(1)

    #     self.surging_after_yaw_180()
    
    def on_exit_yaw_180(self):
        print("Stopping Yaw")
        self.m.set_thrust(DoF.YAW, 0)

    def on_enter_surging_forward_2(self):
        self.led_publisher.publish(144)
        print("Surging forward again")
        self.m.set_thrust(DoF.SURGE, SURGE_THRUST)

        if self.timer_2 is None:
            self.timer_2 = threading.Thread(target=self.timer_async_2, daemon=True)
            self.timer_2.start()
                  
    def on_exit_surging_forward_2(self):
        print("Stopping Surge 2.")
        self.m.set_thrust(DoF.SURGE, 0)
    
    def on_enter_finished(self):
        self.led_publisher.publish(63)
        print("Finished")
        self.m.set_control_mode(DoF.YAW, ControlMode.OPEN_LOOP)
        self.m.set_control_mode(DoF.HEAVE, ControlMode.OPEN_LOOP)
        self.disable_pitch_pid()


    def enable_pitch_pid(self):
        self.m.set_pid_constants(
            DoF.PITCH, PITCH_KP, PITCH_KI, PITCH_KD, PITCH_ACCEPTABLE_ERROR
        )
        self.m.set_target_point(DoF.PITCH, PITCH_TARGET)
        self.m.set_control_mode(DoF.PITCH, ControlMode.CLOSED_LOOP)
    
    def disable_pitch_pid(self):
        self.m.set_control_mode(DoF.PITCH, ControlMode.OPEN_LOOP)

    def timer_async(self):
        time.sleep(ASYNC_CLOCK_DURATION)
        self.yaw_180_after_surging()
    
    def timer_async_2(self):
        print("Timer 2 started")
        time.sleep(ASYNC2_CLOCK_DURATION)
        self.finish()
        
    def timer_async_yaw_manual(self):
        print("Timer yaw started")
        time.sleep(ASYNC_YAW_CLOCK_DURATION)
        self.surging_after_yaw_180()
    
if __name__ == "__main__":
    task = QualificationTask()
    rospy.spin()

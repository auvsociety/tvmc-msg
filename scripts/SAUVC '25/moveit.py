#!/usr/bin/env python3
import time
import threading

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from statemachine import StateMachine, State
from tvmc import MotionController, DoF, ControlMode

from rose_tvmc_msg.msg import LEDControl
from PID_CONSTANTS import *


YAW_THRUST = 60
ASYNC_ROTATION_DURATION = 2.5
ASYNC_CLOCK_DURATION = 10
ASYNC2_CLOCK_DURATION = 9
SURGE_THRUST = 85
ROTATION_YAW = 180

# globals
CURRENT_YAW = 0
START_YAW = 0
REVERSE_YAW = -1


class QualificationTask(StateMachine):
    wait_to_start = State(initial=True)
    initializing_sensors = State()
    fixing_yaw = State()
    enabling_heave_pid = State()
    surging_forward = State()
    yaw_180 = State()
    surging_forward_2 = State()
    finished = State(final=True)


    start_initializing_sensors = wait_to_start.to(initializing_sensors)
    fix_yaw = initializing_sensors.to(fixing_yaw)
    heave_down = fixing_yaw.to(enabling_heave_pid)
    surge_forward = enabling_heave_pid.to(surging_forward)
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
        self.timer_rotate = None
        self.led_publisher = rospy.Publisher("/control/led", LEDControl, queue_size=1)

        super(QualificationTask, self).__init__()

    def on_enter_wait_to_start(self):
        print("Waiting to start.")
        self.m.start()
        time.sleep(0)
        self.start_initializing_sensors()

    def on_depth(self, depth: Float32):

        self.current_depth = depth.data
        
        self.m.set_current_point(DoF.HEAVE, depth.data)

    def set_yaw(self, angle):
        self.yaw_lock = angle
        self.m.set_target_point(DoF.YAW, angle)

    def on_orientation(self, vec: Vector3):
        self.current_yaw = vec.z

        # Set current yaw in the motion controller
        self.m.set_current_point(DoF.YAW, vec.z)

    def on_enter_initializing_sensors(self):
        print("Initializing Sensors.")

        self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
        self.orientation_sub = rospy.Subscriber(
            f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation
        )
        self.depth_sub = rospy.Subscriber(f"/{DATA_SOURCE}/depth", Float32, self.on_depth)


        
           
        time.sleep(5)

        current_time = time.time()
        samples = [120]

        # while self.current_yaw is None:
        #     time.sleep(0)
        self.yaw_lock = 121

        # while time.time() - current_time < 5:
        #     samples.append(self.current_yaw)
        #     time.sleep(0.1)
        #samples = [(360 - x if x >= 180 else x) for x in samples]
        #print(samples)
        #self.yaw_lock = (sum(samples) / len(samples))
        #self.yaw_lock = self.yaw_lock + 360 if self.yaw_lock < 0 else self.yaw_lock
        print("Yaw locked at: ", self.yaw_lock)

        self.fix_yaw()
        # self.surge_forward_directly()

    def on_enter_fixing_yaw(self):
        print("Attempting to fix yaw.")
        self.set_yaw(self.yaw_lock)
        self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)

        while (self.current_yaw is None or abs(self.current_yaw - self.yaw_lock) > YAW_ACCEPTABLE_ERROR):
            time.sleep(0.1)
        
        # self.set_yaw(self.yaw_lock)
        # self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)

        self.heave_down()
    
    def on_enter_yaw_180(self):
        print("Attempting to yaw 180.")

        # time.sleep(1)
        # self.set_yaw(self.yaw_lock - 180)
        # self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)
        
        time.sleep(1.5)
        self.set_yaw(self.yawlock + 180)
        self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)
        

        


        # -------------------------------------------------------------

        # target_yaw = (self.current_yaw + ROTATION_YAW) % 360
        # self.m.set_control_mode(DoF.YAW, ControlMode.OPEN_LOOP)

        # self.m.set_thrust(DoF.YAW,YAW_THRUST)

        while (self.current_yaw is None or abs(self.current_yaw - self.yawlock + 180)> YAW_ACCEPTABLE_ERROR):
            time.sleep(0.1)
        time.sleep(1)
        # self.m.set_thrust(DoF.YAW, 0)
        # -------------------------------------------------------------

        self.surging_after_yaw_180()
        
        
    def on_exit_yaw_180(self):
        print("Stopping Yaw")
        self.m.set_thrust(DoF.YAW, 0)
    	
    
    def on_enter_enabling_yaw_pid(self):
        print("Enabling yaw PID.")
        self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
        self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)
        self.enabling_heave_pid()

    def on_enter_enabling_heave_pid(self):
        print("Enabling heave PID.")
        self.enable_pitch_pid()
        self.m.set_pid_constants(
            DoF.HEAVE,
            HEAVE_KP,
            HEAVE_KI,
            HEAVE_KD,
            HEAVE_ACCEPTABLE_ERROR,
            HEAVE_OFFSET,
        )
        self.m.set_pid_limits(DoF.HEAVE, -10, 10, -25, 25)
        self.m.set_control_mode(DoF.HEAVE, ControlMode.CLOSED_LOOP)
        self.m.set_target_point(DoF.HEAVE, HEAVE_TARGET)

        while (self.current_depth is None or abs(self.current_depth - HEAVE_TARGET) > HEAVE_ACCEPTABLE_ERROR * 3):
            time.sleep(0.1)
            #break

        self.surge_forward()

    def enable_pitch_pid(self):
        self.m.set_pid_constants(
            DoF.PITCH, PITCH_KP, PITCH_KI, PITCH_KD, PITCH_ACCEPTABLE_ERROR
        )
        self.m.set_target_point(DoF.PITCH, PITCH_TARGET)
        self.m.set_control_mode(DoF.PITCH, ControlMode.CLOSED_LOOP)
    
    def disable_pitch_pid(self):
        self.m.set_control_mode(DoF.PITCH, ControlMode.OPEN_LOOP)

    def enable_yaw_pid(self):
        self.m.set_pid_constants(
            DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR
        )
        self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)

    def enable_heave_pid(self):
        self.m.set_pid_constants(
            DoF.HEAVE, HEAVE_KP, HEAVE_KI, HEAVE_KD, HEAVE_ACCEPTABLE_ERROR, HEAVE_OFFSET
        )
        self.m.set_pid_limits(DoF.HEAVE, -10, 10, -25, 25)
        self.m.set_control_mode(DoF.HEAVE, ControlMode.CLOSED_LOOP)
    
    def disable_heave_pid(self):
        self.m.set_control_mode(DoF.HEAVE, ControlMode.CLOSED_LOOP)
    
    def timer_async(self):
        print("Timer 1 started")
        time.sleep(ASYNC_CLOCK_DURATION)
        
        self.yaw_180_after_surging()

    def timer_async_2(self):
        print("Timer 2 started")
        time.sleep(ASYNC2_CLOCK_DURATION)

        self.finish()

    def timer_async_rotate(self):
        print("Timer Started")
        time.sleep(ASYNC_ROTATION_DURATION)

        self.surging_after_yaw_180()

    def on_enter_surging_forward(self):
        #self.enable_pitch_pid()
        time.sleep(1)

        print("Surging forward")
        self.m.set_thrust(DoF.SURGE, SURGE_THRUST)

        if self.timer is None:
            self.timer = threading.Thread(target=self.timer_async, daemon=True)
            self.timer.start()

    def on_enter_surging_forward_2(self):
        # self.enable_pitch_pid()
        time.sleep(0.1)


        print("Surging forward again")
        self.m.set_thrust(DoF.SURGE, SURGE_THRUST)

        if self.timer_2 is None:
            self.timer_2 = threading.Thread(target=self.timer_async_2, daemon=True)
            self.timer_2.start()

    def on_exit_surging_forward(self):
        print("Stopping Surge.")
        self.m.set_thrust(DoF.SURGE, 0)
        # self.disable_pitch_pid()

        self.m.set_control_mode(DoF.YAW, ControlMode.OPEN_LOOP)
        #self.m.set_thrust(DoF.YAW, 50)
        #time.sleep(3)
        #self.m.set_thrust(DoF.YAW, 0)
        #self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)
                  
    def on_exit_surging_forward_2(self):
        print("Stopping Surge 2.")
        self.m.set_thrust(DoF.SURGE, 0)
        # self.disable_pitch_pid()
    
    def on_enter_finished(self):
        self.disable_heave_pid()
        self.disable_pitch_pid()
        


if __name__ == "__main__":
    task = QualificationTask()
    rospy.spin()

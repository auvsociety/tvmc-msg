#!/usr/bin/env python3
import time
import threading

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
from statemachine import StateMachine, State
from tvmc import MotionController, DoF, ControlMode

from rose_tvmc_msg.msg import LEDControl
import collections

# globals
CURRENT_YAW = 0
START_YAW = 0
REVERSE_YAW = -1
DATA_SOURCE = "sensors"

HEAVE_KP = -160
HEAVE_KI = 0
HEAVE_KD = -30
HEAVE_TARGET = 0.5
HEAVE_ACCEPTABLE_ERROR = 0.01
HEAVE_OFFSET = -4

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

YAW_KP = 2
YAW_KI = 0.13
YAW_KD = 0.5
YAW_TARGET = 0
YAW_ACCEPTABLE_ERROR = 1.5


class QualificationTask(StateMachine):
    wait_to_start = State(initial=True)
    initializing_sensors = State()
    fixing_yaw = State()
    enabling_heave_pid = State()
    surging_forward = State()
    finished = State()


    start_initializing_sensors = wait_to_start.to(initializing_sensors)
    fix_yaw = initializing_sensors.to(fixing_yaw)
    heave_down = fixing_yaw.to(enabling_heave_pid)
    surge_forward = enabling_heave_pid.to(surging_forward)
    finish = surging_forward.to(finished)


    def __init__(self):
        self.m = MotionController()
        self.yaw_lock = None
        self.orientation_sub = None
        self.depth_sub = None
        self.current_yaw = None
        self.current_depth = None
        self.timer = None
        self.led_publisher = rospy.Publisher("/diagnostics/led", LEDControl, queue_size=1)
        self.led_message = LEDControl()
        self.led_message.R = -1
        self.led_message.G = -1
        self.led_message.B = -1
        self.led_message.led = 1
        self.depth_led_done = False

        super(QualificationTask, self).__init__()

    def on_enter_wait_to_start(self):
        print("Waiting to start.")
        self.m.start()
        time.sleep(0)
        self.start_initializing_sensors()

    def on_depth(self, depth: Float32):
        if not self.depth_led_done:
            self.depth_led_done = True
            self.led_message.B  = -1
            self.led_publisher.publish(self.led_message)

        self.current_depth = depth.data
        print(self.current_depth)
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
        self.led_message.R = -1
        self.led_message.G = 0
        self.led_message.B = -1
        self.led_publisher.publish(self.led_message)

        self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
        self.orientation_sub = rospy.Subscriber(
            f"/{DATA_SOURCE}/orientation", Vector3, self.on_orientation
        )
        self.depth_sub = rospy.Subscriber(f"/{DATA_SOURCE}/depth", Float32, self.on_depth)

        self.led_publisher.publish(self.led_message)

        self.led_message.R = -1
        self.led_message.B = 0
        self.led_message.G = 0
        self.led_publisher.publish(self.led_message)

        while self.current_depth is None or self.current_depth < 0.1:
           time.sleep(0)
        
        self.led_message.R = 3
        self.led_message.B = 3
        self.led_message.G = -1
        self.led_publisher.publish(self.led_message)
           
        time.sleep(5)

        self.led_message.R = 3
        self.led_message.B = -1
        self.led_publisher.publish(self.led_message)

        current_time = time.time()
        samples = []

        while self.current_yaw is None:
            time.sleep(0)

        while time.time() - current_time < 5:
            samples.append(self.current_yaw)
            time.sleep(0.1)
        samples = [(360 - x if x >= 180 else x) for x in samples]
        print(samples)
        self.yaw_lock = (sum(samples) / len(samples))
        self.yaw_lock = self.yaw_lock + 360 if self.yaw_lock < 0 else self.yaw_lock
        print("Yaw locked at: ", self.yaw_lock)

        self.led_message.R = 0
        self.led_message.G = 1
        self.led_publisher.publish(self.led_message)
        self.fix_yaw()

    def on_enter_fixing_yaw(self):
        print("Attempting to fix yaw.")
        self.set_yaw(self.yaw_lock)
        self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)
        self.heave_down()

    def on_enter_enabling_yaw_pid(self):
        print("Enabling yaw PID.")
        self.m.set_pid_constants(DoF.YAW, YAW_KP, YAW_KI, YAW_KD, YAW_ACCEPTABLE_ERROR)
        self.m.set_control_mode(DoF.YAW, ControlMode.CLOSED_LOOP)
        self.enabling_heave_pid()

    def on_enter_enabling_heave_pid(self):
        print("Enabling heave PID.")
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

        while (
            self.current_depth is None
            or abs(self.current_depth - HEAVE_TARGET) > HEAVE_ACCEPTABLE_ERROR * 3
        ):
            time.sleep(0.1)

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
    
    def timer_async(self):
        time.sleep(35)
        self.finish()

    def on_enter_surging_forward(self):
        print("Surging forward")
        self.m.set_thrust(DoF.SURGE, 75)
        # self.enable_pitch_pid()

        if self.timer is None:
            self.timer = threading.Thread(target=self.timer_async, daemon=True)
            self.timer.start()

    def on_exit_surging_forward(self):
        print("Stopping Surge.")
        self.m.set_thrust(DoF.SURGE, 0)
        # self.disable_pitch_pid()
    
    def on_finished(self):
        self.m.set_control_mode(DoF.HEAVE, ControlMode.OPEN_LOOP)


if __name__ == "__main__":
    task = QualificationTask()
    rospy.spin()

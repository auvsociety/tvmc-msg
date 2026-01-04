"""
BiBi-Sync based Motion Controller
Replaces ROS-based motion_controller.py
"""

import bibi_sync
import struct
from threading import Thread
from time import sleep
from .enums import DoF, Command, ControlMode

# BiBi-Sync topic names (matching TVMC C++ bridge)
TOPICS = {
    "command": "/tvmc/command",
    "control_mode": "/tvmc/control_mode", 
    "current_point": "/tvmc/current_point",
    "target_point": "/tvmc/target_point",
    "pid_constants": "/tvmc/pid_constants",
    "pid_limits": "/tvmc/pid_limits",
    "multi_thrust": "/tvmc/multi_thrust",
    "thrust": "/tvmc/thrust",
    "pwm": "/tvmc/pwm",
    # Sensor topics (from STM32)
    "imu": "/stm32/imu",
    "orientation": "/stm32/orientation",
    "depth": "/stm32/depth",
}

# Message struct sizes
MSG_SIZES = {
    "command": 1,           # uint8
    "control_mode": 2,      # uint8 dof + uint8 mode
    "current_point": 5,     # uint8 dof + float32
    "target_point": 5,      # uint8 dof + float32
    "pid_constants": 21,    # uint8 dof + 5*float32
    "pid_limits": 17,       # uint8 dof + 4*float32
    "multi_thrust": 24,     # 6*float32
    "thrust": 24,           # 6*float32 (per thruster)
    "pwm": 24,              # 6*int32
    "imu": 36,              # 9*float32
    "orientation": 12,      # 3*float32
    "depth": 4,             # float32
}


class MotionController:
    def __init__(self) -> None:
        # init control modes
        self.controlModes = dict.fromkeys(DoF, ControlMode.OPEN_LOOP)
        
        # create BiBi-Sync registry
        self._registry = bibi_sync.PyBibiRegistry()
        
        # create topics for publishing commands to TVMC
        self._command_topic = self._registry.get_byte_topic(TOPICS["command"], 8)
        self._control_mode_topic = self._registry.get_byte_topic(TOPICS["control_mode"], 8)
        self._current_point_topic = self._registry.get_byte_topic(TOPICS["current_point"], 8)
        self._target_point_topic = self._registry.get_byte_topic(TOPICS["target_point"], 8)
        self._pid_constants_topic = self._registry.get_byte_topic(TOPICS["pid_constants"], 8)
        self._pid_limits_topic = self._registry.get_byte_topic(TOPICS["pid_limits"], 8)
        self._multi_thrust_topic = self._registry.get_byte_topic(TOPICS["multi_thrust"], 8)
        
        # sensor topics (for receiving from STM32 via UART bridge)
        self._imu_topic = self._registry.get_byte_topic(TOPICS["imu"], 16)
        self._orientation_topic = self._registry.get_byte_topic(TOPICS["orientation"], 16)
        self._depth_topic = self._registry.get_byte_topic(TOPICS["depth"], 16)
        
        # PWM output topic (for sending to STM32)
        self._pwm_topic = self._registry.get_byte_topic(TOPICS["pwm"], 8)
        
        self._running = False
        
    def start(self) -> None:
        """Start the motion controller (no ROS nodes to launch)"""
        self._running = True
        sleep(0.1)  # small delay for initialization
        
    def stop(self) -> None:
        """Stop the motion controller"""
        self._running = False
        self.send_command(Command.SHUT_DOWN)

    def set_thrust(self, dof: DoF, thrust: float) -> None:
        """Set thrust for a single DoF (legacy compatibility)"""
        # Build multi-thrust with only this DoF set
        thrusts = [0.0] * 6
        thrusts[dof.value] = thrust
        self.set_multi_thrust(*thrusts)

    def set_multi_thrust(self, surge=0.0, sway=0.0, heave=0.0, 
                         roll=0.0, pitch=0.0, yaw=0.0) -> None:
        """Set thrust for all DoFs in open loop"""
        # Check for closed loop modes
        for dof, value in [(DoF.SURGE, surge), (DoF.SWAY, sway), (DoF.HEAVE, heave), 
                           (DoF.ROLL, roll), (DoF.PITCH, pitch), (DoF.YAW, yaw)]:
            if self.controlModes[dof] == ControlMode.CLOSED_LOOP and value != 0:
                raise AssertionError(f"Cannot set thrust for DoF {dof.name} in closed loop mode.")
        
        # Pack as 6 floats: surge, sway, heave, roll, pitch, yaw
        data = struct.pack('ffffff', surge, sway, heave, roll, pitch, yaw)
        self._multi_thrust_topic.publish(data)

    def set_control_mode(self, dof: DoF, control: ControlMode) -> None:
        """Set control mode for a DoF"""
        # Pack as: uint8 dof + uint8 mode
        data = struct.pack('BB', dof.value, control.value)
        self._control_mode_topic.publish(data)
        self.controlModes[dof] = control

    def send_command(self, command: Command) -> None:
        """Send a command to TVMC"""
        data = struct.pack('B', command.value)
        self._command_topic.publish(data)

    def set_current_point(self, dof: DoF, current_point: float) -> None:
        """Set current point for PID controller"""
        # Pack as: uint8 dof + float32 current
        data = struct.pack('Bf', dof.value, current_point)
        self._current_point_topic.publish(data)

    def set_pid_constants(self, dof: DoF, kp: float, ki: float, kd: float, 
                          acceptable_error: float, ko: float = 0.0) -> None:
        """Set PID constants for a DoF"""
        # Pack as: uint8 dof + 5*float32
        data = struct.pack('Bfffff', dof.value, kp, ki, kd, acceptable_error, ko)
        self._pid_constants_topic.publish(data)

    def set_pid_limits(self, dof: DoF, output_min: float, output_max: float,
                       integral_min: float, integral_max: float) -> None:
        """Set PID limits for a DoF"""
        # Pack as: uint8 dof + 4*float32
        data = struct.pack('Bffff', dof.value, output_min, output_max, 
                          integral_min, integral_max)
        self._pid_limits_topic.publish(data)

    def set_target_point(self, dof: DoF, target: float) -> None:
        """Set target point for PID controller"""
        # Pack as: uint8 dof + float32 target
        data = struct.pack('Bf', dof.value, target)
        self._target_point_topic.publish(data)

    # Sensor reading methods
    def get_orientation(self):
        """Get latest orientation (roll, pitch, yaw) from STM32"""
        result = self._orientation_topic.peek_latest()
        if result:
            data, epoch = result
            if len(data) >= 12:
                roll, pitch, yaw = struct.unpack('fff', bytes(data[:12]))
                return (roll, pitch, yaw, epoch)
        return None

    def get_depth(self):
        """Get latest depth from STM32"""
        result = self._depth_topic.peek_latest()
        if result:
            data, epoch = result
            if len(data) >= 4:
                depth = struct.unpack('f', bytes(data[:4]))[0]
                return (depth, epoch)
        return None

    def get_imu(self):
        """Get latest IMU data from STM32"""
        result = self._imu_topic.peek_latest()
        if result:
            data, epoch = result
            if len(data) >= 36:
                values = struct.unpack('fffffffff', bytes(data[:36]))
                return {
                    'accel': (values[0], values[1], values[2]),
                    'gyro': (values[3], values[4], values[5]),
                    'mag': (values[6], values[7], values[8]),
                    'epoch': epoch
                }
        return None

    def send_pwm(self, pwm_values: list) -> None:
        """Send PWM values directly to STM32 (bypasses TVMC)"""
        if len(pwm_values) != 6:
            raise ValueError("PWM values must be a list of 6 integers")
        data = struct.pack('iiiiii', *pwm_values)
        self._pwm_topic.publish(data)

    def __del__(self):
        if hasattr(self, '_running') and self._running:
            self.stop()

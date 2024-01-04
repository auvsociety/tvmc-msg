import rose_tvmc_msg.msg as msg
from enum import IntEnum


class DoF(IntEnum):
    SURGE = msg.DoF.SURGE
    SWAY = msg.DoF.SWAY
    HEAVE = msg.DoF.HEAVE
    YAW = msg.DoF.YAW
    PITCH = msg.DoF.PITCH
    ROLL = msg.DoF.ROLL


class ControlMode(IntEnum):
    OPEN_LOOP = msg.ControlMode.OPEN_LOOP
    CLOSED_LOOP = msg.ControlMode.CLOSED_LOOP


class Command(IntEnum):
    RESET_THRUSTERS = msg.Command.RESET_THRUSTERS
    REFRESH = msg.Command.REFRESH
    SHUT_DOWN = msg.Command.SHUT_DOWN

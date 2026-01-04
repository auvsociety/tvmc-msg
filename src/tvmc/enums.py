"""
BiBi-Sync TVMC Enums
Pure Python enums without ROS dependency
"""

from enum import IntEnum


class DoF(IntEnum):
    """Degrees of Freedom"""
    SURGE = 0
    SWAY = 1
    HEAVE = 2
    ROLL = 3
    PITCH = 4
    YAW = 5


class ControlMode(IntEnum):
    """Control modes for each DoF"""
    CLOSED_LOOP = 0
    OPEN_LOOP = 1


class Command(IntEnum):
    """Commands for motion controller"""
    RESET_THRUSTERS = 0
    REFRESH = 1
    SHUT_DOWN = 2

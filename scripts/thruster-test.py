#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int32MultiArray
import blessings
from sshkeyboard import listen_keyboard, stop_listening

PWM_OFF = 1472
PWM_ON = 1600
NUM_THRUSTERS = 7

term = blessings.Terminal()

pub = rospy.Publisher('/control/pwm', Int32MultiArray, queue_size=10)
msg = Int32MultiArray()


def test_thruster(thruster: int):
    data = [PWM_OFF] * NUM_THRUSTERS
    
    if thruster:
        data[thruster - 1] = PWM_ON
    
    msg.data = data
    pub.publish(msg)


def on_press(key):
    if key == 'q':
        stop_listening()

    if key.isdigit() and 1 <= int(key) <= NUM_THRUSTERS:
        print(f"\nTesting thruster {int(key)}.")
        test_thruster(int(key))
    else:
        print("Stopping thrusters.")
        test_thruster(0)


def on_release(key):
    print("Stopping thrusters.")
    test_thruster(0)


if __name__ == '__main__':
    print(term.fullscreen())
    print(term.clear())

    rospy.init_node('pwm_tester', anonymous=True)
    print(f"Press (1-{NUM_THRUSTERS}) to test thruster. Q to exit.\n")
    
    listen_keyboard(on_press=on_press, on_release=on_release)

    print(term.clear())
    exit(0)

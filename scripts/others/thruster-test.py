#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int32MultiArray
import blessings
from sshkeyboard import listen_keyboard, stop_listening
import json
import os

#pwm values based on t200 thruster specs
PWM_NEUTRAL = 1500
PWM_TEST = 1600

NUM_THRUSTERS = 6

def load_thruster_count():
    global NUM_THRUSTERS
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths = [
        os.path.join(script_dir, '..', '..', '..', 'tvmc', 'config', 'config.json'),
    ]
    
    for config_path in config_paths:
        config_path = os.path.normpath(config_path)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    NUM_THRUSTERS = config.get('thrusterSpec', {}).get('noOfThrusters', NUM_THRUSTERS)
                    rospy.loginfo(f"Loaded thruster count from config: {NUM_THRUSTERS}")
                    return
            except (json.JSONDecodeError, IOError) as e:
                rospy.logwarn(f"Could not parse config at {config_path}: {e}")
    
    rospy.logwarn(f"Config not found, using default thruster count: {NUM_THRUSTERS}")

term = blessings.Terminal()

pub = None
msg = Int32MultiArray()


def test_thruster(thruster: int):
    data = [PWM_NEUTRAL] * NUM_THRUSTERS
    
    if thruster and 1 <= thruster <= NUM_THRUSTERS:
        data[thruster - 1] = PWM_TEST
    
    msg.data = data
    pub.publish(msg)


def stop_all_thrusters():
    msg.data = [PWM_NEUTRAL] * NUM_THRUSTERS
    pub.publish(msg)


def on_press(key):
    if key == 'q':
        stop_all_thrusters()
        stop_listening()
        return

    if key.isdigit():
        thruster_num = int(key)
        if 1 <= thruster_num <= NUM_THRUSTERS:
            print(f"\nTesting thruster {thruster_num}. PWM: {PWM_TEST}")
            test_thruster(thruster_num)
        else:
            print(f"Invalid thruster number. Use 1-{NUM_THRUSTERS}.")
            stop_all_thrusters()
    else:
        print("Stopping thrusters.")
        stop_all_thrusters()


def on_release(key):
    if key != 'q':
        print("Stopping thrusters.")
        stop_all_thrusters()


if __name__ == '__main__':
    print(term.fullscreen())
    print(term.clear())

    rospy.init_node('pwm_tester', anonymous=True)
    load_thruster_count()
    pub = rospy.Publisher('/control/pwm', Int32MultiArray, queue_size=10)
    rospy.sleep(0.5)
    stop_all_thrusters()
    
    print(term.bold("Thruster Tester"))
    print(f"Thrusters configured: {NUM_THRUSTERS}")
    print(f"PWM Neutral: {PWM_NEUTRAL}, PWM Test: {PWM_TEST}")
    print(f"\nPress (1-{NUM_THRUSTERS}) to test thruster. Q to exit.\n")
    
    listen_keyboard(on_press=on_press, on_release=on_release)
    stop_all_thrusters()
    
    print(term.clear())
    print("Bye-bye!")
    exit(0)

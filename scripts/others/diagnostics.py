#!/usr/bin/env python3
import rospy
from rose_tvmc_msg.msg import LEDControl
from std_msgs.msg import Int16
import time


def calculate_current_state(state):
    if state < 0:
        return 0
    
    if state == 0:
        return 1
    
    # get current state
    # time = current time in ms
    # duration in ms = 1000 // rate
    # state = on if time // duration is odd, else off

    return (int(time.time() * 1000 * state) // 1000) & 1


class LEDController:
    def __init__(self, init_node=False, rate=10):
        if init_node:
            rospy.init_node("led_controller")

        self.publisher = rospy.Publisher("/control/led", Int16, queue_size=1)
        self.subscriber = rospy.Subscriber("/diagnostics/led", LEDControl, self.on_led, queue_size=1)
        self.rate = rospy.Rate(rate)
        self.state = {}
    
    def on_led(self, msg):
        self.state[msg.led] = (msg.R, msg.G, msg.B)

    def publish(self):
        number = 0

        for led in self.state:
            r, g, b = (calculate_current_state(x) for x in self.state[led])
            num = b << 2 | g << 1 | r
            number |= num << (led * 3)
        
        self.publisher.publish(number)
    
    def start(self):
        while not rospy.is_shutdown():
            self.publish()
            self.rate.sleep()


def main():
    rospy.init_node("diagnostics")
    controller = LEDController()
    controller.start()


if __name__ == "__main__":
    main()
            


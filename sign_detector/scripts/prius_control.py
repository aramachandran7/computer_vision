#!/usr/bin/env python3

import rospy
from std_msgs.msg import Header, String
from geometry_msgs.msg import PoseArray
import numpy as np
from prius_msgs.msg import Control
from sign_tracker_node import Sign


"""
rostopic pub /prius prius_msgs/Control "header:
  seq: 0
  stamp: {secs: 0, nsecs: 0}
  frame_id: ''
throttle: 2.0
brake: 0.0
steer: 0.0
shift_gears: 0"
"""
class PriusControl(object):
    """docstring for PriusControl."""

    def __init__(self):
        rospy.init_node("PriusControl")
        rospy.Subscriber("/sign_detected", )
        self.prius_control_publisher = rospy.Publisher("/prius", Control, queue_size=10)
        rospy.Subscriber("sign_position", PoseArray, self.sign_callback)
        self.signs = []

    def drive(self):
        """
        return control object
        """
        
        my_control = Control(header=Header(), throttle= ,brake= , steer= ,shift_gears= )
        self.prius_control_publisher.Publish(my_control)

    def sign_callback(self, msg):
        """
        grab sign positions
        """
        signs = msg.poses


    def run(self):
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            r.sleep()
            pass


if __name__ == '__main__':
    PC = PriusControl()
    PC.run()

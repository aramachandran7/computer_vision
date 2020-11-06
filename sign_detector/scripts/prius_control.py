#!/usr/bin/env python3

import rospy
from std_msgs.msg import Header, String
import numpy as np
from prius_msgs.msg import Control


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
        self.priusPub = rospy.Publisher("/prius", Control, queue_size=10)

    def drive(self):
        """
        return control object
        """
        control = Control(Header())

    def run(self):
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            r.sleep()   
            pass


if __name__ == '__main__':
    PC = PriusControl()
    PC.run()

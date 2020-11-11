#!/usr/bin/env python3

import rospy
from std_msgs.msg import Header, String
from geometry_msgs.msg import PoseArray
import numpy as np
from prius_msgs.msg import Control
from sign_detector.msg import SignMatch


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
    """control your prius"""

    def __init__(self):
        rospy.init_node("PriusControl")
        self.prius_control_publisher = rospy.Publisher("/prius", Control, queue_size=10)
        rospy.Subscriber("/sign_position", SignMatch, self.sign_callback)
        rospy.on_shutdown(self.stop_the_fucking_car)
        self.sign_detected = (0,0)
        self.control = None

        self.normal_throttle = 0.1
        self.normal_brake = 0.0
        self.normal_gears = 2

        self.area_threshold_cut_throttle = 50 # minimum area for throttle cut
        self.area_threshold_apply_brakes = 250 # minimum area to apply brakes
        self.brake_constant = 3000.0



    def drive(self):
        """
        return control object
        """
        if self.sign_detected == (0,0):
            rospy.loginfo("No sign detected, should be normal operation")
            # in the case where there is no sign, set normal values
            self.control = Control(header=Header(stamp=rospy.Time.now(),frame_id=''), throttle= self.normal_throttle,brake= self.normal_brake, steer=0 ,shift_gears=self.normal_gears)

        else:
            rospy.loginfo("sign detected")
            area = self.sign_detected[1]
            matches = self.sign_detected[0]
            if self.control.brake == 1.0 and  
            if matches>0:
                # end all be all
                rospy.loginfo("Many matches, braking hard")
                self.control = Control(header=Header(stamp=rospy.Time.now(),frame_id=''), throttle=0 ,brake=1.0 , steer=0 ,shift_gears=self.normal_gears )
            else:
                # handle the case where there is a sign
                rospy.loginfo("No matches, but area")
                if area > self.area_threshold_cut_throttle:
                    rospy.loginfo("cutting throttle")
                    self.control = Control(header=Header(stamp=rospy.Time.now(),frame_id=''), throttle=0 ,brake=0 , steer=0 ,shift_gears=self.normal_gears)
                elif area > self.area_threshold_apply_brakes:
                    rospy.loginfo("applying brake")
                    self.control = Control(header=Header(stamp=rospy.Time.now(),frame_id=''), throttle=0 ,brake= 1.0, steer=0 ,shift_gears=self.normal_gears )

        self.prius_control_publisher.publish(self.control)

    def sign_callback(self, msg):
        """
        grab sign positions
        """
        self.sign_detected = (msg.matches, msg.area)

    def stop_the_fucking_car(self):
        rospy.loginfo("stopping the fucking car")
        self.control = Control(header=Header(stamp=rospy.Time.now(),frame_id=''), throttle=0 ,brake=1.0 , steer=0 ,shift_gears=self.normal_gears )
        self.prius_control_publisher.publish(self.control)

    def run(self):
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.drive()
            r.sleep()



if __name__ == '__main__':
    PC = PriusControl()
    PC.run()

"""
Objective - detect signs with shape and color
pass image into cv

to try:
blur image
"""
import rospy
import cv2
import numpy as np

class Detector(object):
    """docstring for Detector."""

    def __init__(self):
        cv2.namedWindow('video_window')
        cv2.namedWindow('threshold_image')
        self.hue_lower_bound = 17
        self.hue_upper_bound = 29
        cv2.createTrackbar('hue lower bound', 'threshold_image',0,255, self.set_hue_lower_bound)
        cv2.createTrackbar('hue upper bound', 'threshold_image',0,255, self.set_hue_upper_bound)

        cv2.setMouseCallback('video_window', self.process_mouse_event)

    def process_mouse_event(self, event, x,y,flags,param):
        """ Process mouse events so that you can see the color values
            associated with a particular pixel in the camera images """
        image_info_window = 255*np.ones((500,500,3))
        cv2.putText(image_info_window,
                    'Color (h=%d,s=%d,v=%d)' % (self.hsv_image[y,x,0], self.hsv_image[y,x,1], self.hsv_image[y,x,2]),
                    (5,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0))

        cv2.imshow('image_info', image_info_window)
        cv2.waitKey(5)

    def set_hue_lower_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the hue lower bound """

        self.hue_lower_bound = val

    def set_hue_upper_bound(self, val):
        """ A callback function to handle the OpenCV slider to select the hue upper bound """

        self.hue_upper_bound = val

    def process_image_stop_sign(self):
        bimage1 = cv2.inRange(self.hsv_image, (150,0,0), (190, 255, 255))
        bimage2 = cv2.inRange(self.hsv_image, (0,0,0),(10,255,255))
        self.binary_image = cv2.bitwise_or(bimage1,bimage2)
        cnts = cv2.findContours(self.binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # print(np.shape(cnts))
        cv2.drawContours(self.cv_image,cnts[0],-1,(0,255,0),3)

    def process_image_rail_road_sign(self):
        self.hsv_image = cv2.GaussianBlur(self.hsv_image,(3,3),cv2.BORDER_DEFAULT) # apply smoothing
        self.binary_image = cv2.inRange(self.hsv_image, (self.hue_lower_bound,150,0), (self.hue_upper_bound, 255, 255))
        cnts = cv2.findContours(self.binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # print(np.shape(cnts))
        cv2.drawContours(self.cv_image,cnts[0],-1,(0,255,0),3)

    def main(self):
        while True:
            self.cv_image = cv2.imread("./SignImages/StopSign2.jpeg",cv2.IMREAD_COLOR)
            self.hsv_image = cv2.cvtColor(self.cv_image,cv2.COLOR_BGR2HSV)
            self.process_image_stop_sign()
            cv2.imshow('video_window', self.cv_image)
            cv2.imshow('threshold_image',self.binary_image)
            cv2.waitKey(5)
# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

if __name__ == '__main__':
    det = Detector()
    det.main()

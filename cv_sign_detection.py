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
        self.threshold_cnt = 1200
        self.red_ranges = [((0,150,0),(10,255,255)),((150,150,0),(190,255,255))]
        self.red_signs = [("stop_sign",8),("yield_sign",3)]
        self.yellow_ranges = [(17, 150, 0), (29, 255, 255)]
        self.yellow_signs = [("rail_road_sign", 0)]

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

    # def process_image(self, range1, range2):
    #
    #     bimage1 = cv2.inRange(self.hsv_image, (150,150,0), (190, 255, 255))
    #     bimage2 = cv2.inRange(self.hsv_image, (0,150,0),(10,255,255))
    #     self.binary_image = cv2.bitwise_or(bimage1,bimage2)
    #     self.make_cnts()



    def process_image_colors(self, ranges, sign_classifications):
        """
        returns stuff
        :param ranges: list of HSV ranges as tuples, ((a,b,c), (a1, b1, c1))
        :param sign_classifications: list of tuples, [(sign_type, number of points)]
        """
        # base black image
        self.binary_image = cv2.inRange(self.hsv_image, (255,255,255),(0,0,0))
        # compounded bitwise_or
        for range in ranges:
            # generate biamge, add to bimages
            self.binary_image = cv2.bitwise_or(self.binary_image, cv2.inRange(self.hsv_image, (range[0]), (range[1])))

        primary_shapes = self.make_cnts()
        list_of_signs = []
        for (index, shape) in enumerate(primary_shapes):
            for sign in sign_classifications:
                if(shape[0] == sign[1]):
                    print("found " + sign[0])
                    list_of_signs.append((sign[0],shape[1]))
                    cv2.circle(self.cv_image, shape[1], 7, (255, 0, 0), -1)

        return list_of_signs

    def process_image_red_sign(self):
        """
        Returns list of tuples [(sign_type, location)]
        encapsulate elsewhere
        """
        bimage1 = cv2.inRange(self.hsv_image, (150,150,0), (190, 255, 255))
        bimage2 = cv2.inRange(self.hsv_image, (0,150,0),(10,255,255))
        self.binary_image = cv2.bitwise_or(bimage1,bimage2)

        primary_shapes = self.make_cnts()
        list_of_signs = []
        for (index, shape) in enumerate(primary_shapes):
            if (shape[0] == 3):
                print('found yield')
                list_of_signs.append(("yield_sign",shape[1]))
            elif shape[0] == 8:
                print('found stop')
                list_of_signs.append(("stop_sign",shape[1]))
            else:
                print('found unknown')

        return list_of_signs





    def make_cnts(self):
        """
        return list of tuples (shape center, vertices) of contours with enough area
        (Based on threshold distance (optimal car stopping distance))
        """
        cnts, hierarchy = cv2.findContours(self.binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        primary_cnts = []
        for cnt in cnts:
            if cv2.contourArea(cnt) > self.threshold_cnt:
                print(cv2.contourArea(cnt))
                epsilon = 0.025*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                # calculate shape center
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                primary_cnts.append((len(approx), (cX, cY)))
                cv2.drawContours(self.cv_image,[approx],0,(0,255,0),3)
        # print("number of vertices in shape: ", len(approx))
        return primary_cnts




    def main(self):
        while True:
            self.cv_image = cv2.imread("./SignImages/RailRoad1.jpg",cv2.IMREAD_COLOR)
            self.hsv_image = cv2.cvtColor(self.cv_image,cv2.COLOR_BGR2HSV)
            print(self.process_image_colors(self.red_ranges,self.red_signs))
            print(self.process_image_colors(self.yellow_ranges,self.yellow_signs))
            cv2.imshow('video_window', self.cv_image)
            cv2.imshow('threshold_image',self.binary_image)
            cv2.waitKey(5)
# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

if __name__ == '__main__':
    det = Detector()
    det.main()

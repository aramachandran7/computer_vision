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
        cv2.namedWindow('region_window')
        cv2.namedWindow('compare_images')
        cv2.namedWindow('template')
        self.hue_lower_bound = 17
        self.hue_upper_bound = 29
        cv2.createTrackbar('hue lower bound', 'threshold_image',0,255, self.set_hue_lower_bound)
        cv2.createTrackbar('hue upper bound', 'threshold_image',0,255, self.set_hue_upper_bound)
        self.threshold_cnt = 800
        self.red_ranges = [((0,150,0),(10,255,255)),((150,150,0),(190,255,255))]
        self.red_signs = [("stop_sign",8),("yield_sign",3)]
        self.yellow_ranges = [((17, 150, 0), (29, 255, 255)), ((0,0,0), (255, 255,50))]
        self.yellow_signs = [("rail_road_sign", 0)]
        # self.orange_ranges = [((0,180,220),(10,255,255)), ((0,0,0), (255, 255,50))]
        self.orange_ranges = [((0,180,220),(10,255,255))]

        self.orange_signs = [("road_sign", 4)]
        self.scenes = {
            "stop_scene": cv2.imread("./SignImages/StopSignScene.jpeg",cv2.IMREAD_COLOR),
            "yield_scene": cv2.imread("./SignImages/StopSignScene.jpeg",cv2.IMREAD_COLOR),
            "railroad_scene": cv2.imread("./SignImages/RailRoadSignScene.jpeg",cv2.IMREAD_COLOR),
            "road_scene": cv2.imread("./SignImages/RoadScene.jpeg",cv2.IMREAD_COLOR),
        }


        self.original_image = None

        self.min_match_count = 10
        self.orb = cv2.ORB_create()

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(index_params,search_params)
        self.bf = cv2.BFMatcher()



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

        # blur binary images
        self.binary_image = cv2.GaussianBlur(self.binary_image,(3,3),0)

        # generate bounding rectangles around color matched contours
        list_of_signs = []
        rect_regions = self.make_cnts()

        print("there are ", len(rect_regions), "rectangular regions")
        # sign_in_region = self.feature_mapper(rect_regions[1])

        # for rect_region in rect_regions:
            # returns boolean, if features between template sign and region match up then return true
        sign_in_region = self.feature_mapper(rect_regions[0])
            # if sign_in_region is not None:
            #     list_of_signs.append((rect_region,sign_in_region))

        return list_of_signs
        # shape detection code
        # for (index, shape) in enumerate(primary_shapes):
        #     for sign in sign_classifications:
        #         if(shape[0] == sign[1]):
        #             print("found " + sign[0])
        #             list_of_signs.append((sign[0],shape[1]))
        #             cv2.circle(self.cv_image, shape[1], 7, (255, 0, 0), -1)
        #
        # return list_of_signs

    def make_cnts(self):
        """
        return list of tuples (shape center, vertices) of contours with enough area
        (Based on threshold distance (optimal car stopping distance))
        """
        cnts, hierarchy = cv2.findContours(self.binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        primary_regions = []
        for cnt in cnts:
            if cv2.contourArea(cnt) > self.threshold_cnt:
                epsilon = 0.025*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                # calculate shape center
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # primary_cnts.append((len(approx), (cX, cY)))
                # draw boudnign rectangle if higher than area
                x,y,w,h = cv2.boundingRect(cnt)
                self.cv_image = cv2.rectangle(self.cv_image,(x,y),(x+w,y+h),(255,0,0),2)
                primary_regions.append((x,y,w,h))
                cv2.drawContours(self.cv_image,[approx],0,(0,255,0),3)
        # print("number of vertices in shape: ", len(approx))
        return primary_regions

    # def feature_mapper(self,rect_region):
    #     [x,y,w,h] = rect_region
    #
    #     img2 = self.original_image[y:y+h,x:x+w]
    #
    #     img1 = cv2.imread("./SignImages/RoadSignScene.jpeg",cv2.IMREAD_COLOR)
    #     img1 = cv2.resize(img1,(img2.shape[1], img2.shape[0]))
    #
    #     # difference1 = cv2.subtract(img1,img2)
    #     # difference2 = cv2.subtract(img2,img1)
    #     # average = difference1.mean(axis=0).mean(axis=0)
    #     # average2 = difference2.mean(axis=0).mean(axis=0)
    #     # print((average+average2)/2)
    #     # cv2.imshow('compare_images',difference1)
    #     img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #     img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #     cv2.imshow('region_window',img2)
    #     cv2.imshow('template',img1)
    #
    #
    #     # find the keypoints and descriptors with SIFT
    #     kp1, des1 = self.orb.detectAndCompute(img1,None)
    #     kp2, des2 = self.orb.detectAndCompute(img2,None)
    #     # FLANN parameters
    #     matches = self.bf.knnMatch(des1,des2,k=2)
    #     good_matches = []
    #     for m,n in matches:
    #         if(m.distance < 0.7 * n.distance):
    #             good_matches.append((m.queryIdx,m.trainIdx))
    #     print("good matches" + str(len(good_matches)))
    #     # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    #     # plt.imshow(img3,),plt.show()

# https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
    def feature_mapper(self,rect_region):
        """
        For every sign in sign dictionary, run SIFT and determine keypoint differences for region of interest.
        Calculate
        """
        (x,y,w,h) = rect_region
        # we want to compare features between a template image and the bounding box
        scene = cv2.imread("./SignImages/StopSignScene.jpeg",cv2.IMREAD_COLOR)
        key = "road_scene"
        region = self.original_image[y:y+h,x:x+w]
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        scene = cv2.cvtColor(scene,cv2.COLOR_BGR2GRAY)
        # region = cv2.resize(region,(scene.shape[0], scene.shape[1]))
        cv2.imshow('template', scene)
        cv2.imshow('region_window', region)
        # keypoint generation with SIFT
        # kp_region = self.orb.detect(region,None)
        # kp_region, des_region = self.orb.compute(region,None)
        kp_region,des_region = self.orb.detectAndCompute(region,None)

        # walk through all template signs
        # for (key, scene) in self.scenes.items():

        # scene = self.scenes["road_scene"]

        kp_template, des_template = self.orb.detectAndCompute(scene, None)
        matches = self.bf.match(des_region, des_template)
        # matches = self.flann.knnMatch(des_template, des_region,k=2)
        print("matches found: ", len(matches))
        # store all the good matches as per Lowe's ratio test.
        # good = []
        # for m,n in matches:
        #     if m.distance < 0.7*n.distance:
        #         good.append(m)
        matches = sorted(matches, key = lambda x:x.distance)

        # img3 = cv2.drawMatches(scene,kp_template,region,kp_region,matches,None, flags=2)
        # cv2.imshow('compare_images', img3)
        # if len(good)>self.min_match_count:
        #     # good chance that features detected between template and search image
        #     cv2.imshow('compare_images', region)
        #     print('Probably an object match, %s matches found' % (len(good)))
        #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        #
        #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #     matchesMask = mask.ravel().tolist()
        #
        #     h,w = scene.shape
        #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #     dst = cv2.perspectiveTransform(pts,M)
        #
        #     region = cv2.polylines(region,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        #
        # else:
        #     print("Not enough matches are found - %s/%s found" % (len(good),self.min_match_count))
        #     matchesMask = None

        #
        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #         singlePointColor = None,
        #         matchesMask = matchesMask, # draw only inliers
        #         flags = 2)
        #
        # bothimages = cv2.drawMatches(scene,kp_template,region,kp_region,good,None,flags=2)

        # cv2.imshow('compare_images', bothimages)



    def main(self):
        while True:
            self.cv_image = cv2.imread("./SignImages/RoadSign1.jpeg",cv2.IMREAD_COLOR)
            self.original_image = self.cv_image.copy()
            self.hsv_image = cv2.cvtColor(self.cv_image,cv2.COLOR_BGR2HSV)
            # print(self.process_image_colors(self.red_ranges,self.red_signs))
            # print(self.process_image_colors(self.yellow_ranges,self.yellow_signs))
            print(self.process_image_colors(self.orange_ranges, self.orange_signs))
            cv2.imshow('video_window', self.cv_image)
            cv2.imshow('threshold_image',self.binary_image)
            cv2.waitKey(5)
# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
# https://github.com/DakotaNelson/robot-street-signs
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
#

if __name__ == '__main__':
    det = Detector()
    det.main()

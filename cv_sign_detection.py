"""
Objective - detect signs with shape and color
pass image into cv

to try:
blur image
"""
# import rospy
import cv2
import numpy as np
import time
class Detector(object):
    """docstring for Detector."""

    def __init__(self):
        # cv2.namedWindow('video_window')
        cv2.namedWindow('threshold_image')
        cv2.namedWindow('new_region')
        # cv2.namedWindow('compare_images')
        cv2.namedWindow('old_region')
        self.hue_lower_bound = 17
        self.hue_upper_bound = 29
        # cv2.createTrackbar('hue lower bound', 'threshold_image',0,255, self.set_hue_lower_bound)
        # cv2.createTrackbar('hue upper bound', 'threshold_image',0,255, self.set_hue_upper_bound)
        self.threshold_cnt = 200
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

        # video processing
        self.cap = None
        self.previous_regions = []
        self.widen_region = 10
        self.previous_frame = None
        self.squareness_threshold = .25



        cv2.setMouseCallback('video_window', self.process_mouse_event)



    def main_video_processor(self):
        """
        Handle video capture loop, pass frames onwards to functions
        """
        # init video capture
        self.cap = cv2.VideoCapture('./SignImages/ApproachingStopSign.mp4')

        while(self.cap.isOpened()):
            ret, frame = self.cap.read()

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = self.process_frame(frame)

            cv2.imshow('frame',display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.02)

        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        """
        return areas of interest from image
        :param frame: video frame, BGR OpenCV numpy array
        """
        # in theory, walk through all ranges and track objects for all color ranges
        # for now, we'll just do it for a stop sign

        display_frame = np.copy(frame) # includes bounding boxes + drawings
        color_binary = self.process_color_ranges(frame, self.red_ranges)
        cv2.imshow('threshold_image',color_binary)

        # area = self.get_min_area(car_speed)
        new_regions = self.get_regions_of_interest(frame=color_binary,min_area=self.threshold_cnt)

        new_regions = self.widen_regions(frame, new_regions)

        for region in new_regions:
             # draw blue
            (x,y,w,h) = region
            cv2.rectangle(display_frame,(x,y),(x+w,y+h),(255,0,0),2)
        for region in self.previous_regions:
            # draw green
            (x,y,w,h) = region
            cv2.rectangle(display_frame,(x,y),(x+w,y+h),(0,255,0),2)


        self.track_regions(frame, new_regions)


        self.previous_regions = new_regions
        self.previous_frame = frame
        return display_frame


    def track_regions(self,frame, new_regions):
        """
        compare new regions to old ones, handle different cases
        calculate a confidence score for every region based on its past history.
        """
        if len(new_regions) > len(self.previous_regions):
            # a new region has been found. Well shit.
            print('new region found!')
            pass
        elif  len(new_regions) < len(self.previous_regions):
            # a region disappeared. Well shit.
            print('region dissappeared')
            pass
        else:
            for (i, coordinates) in enumerate(new_regions):
                # perform some comparison and tracking
                # subtraction? Feature mapping?
                (x,y,w,h) = coordinates
                (x1,y1,w1,h1) = self.previous_regions[i]

                region_new = frame[y:y+h, x:x+h]
                region_old = self.previous_frame[y1:y1+h1, x1:x1+h1]

                num_matches = self.feature_mapper(region_new, region_old)
                print(num_matches)
                # quantify confidence based on num matches




    def widen_regions(self, frame, regions):
        """
        for every region in a list of tuple rectangle dimensions, if region can be widened, widen it
        """
        new_regions = []
        for region in regions:
            (x,y,w,h) = region
            if x>self.widen_region and y> self.widen_region and (frame.shape[1]-x) > self.widen_region and (frame.shape[0]-y)>self.widen_region:
                region_wide = (x-self.widen_region, y-self.widen_region, w+2*self.widen_region, h+2*self.widen_region)
                new_regions.append(region_wide)
            else:
                new_regions.append(region)
        return new_regions


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


    def process_color_ranges(self, frame, ranges):
        """
        Simply processes a frame for given HSV ranges
        """
        hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        binary_image = cv2.inRange(hsv_frame, (255,255,255),(0,0,0))
        for range in ranges:
            # generate biamge, use compounded bitwise_or to add to binary_image
            binary_image = cv2.bitwise_or(binary_image, cv2.inRange(hsv_frame, (range[0]), (range[1])))

        # blur binary images
        binary_image = cv2.GaussianBlur(binary_image,(3,3),0)

        return binary_image

    def get_regions_of_interest(self, frame, min_area):
        """
        return list of tuple rectangluar regions of interest in an image based on contour mapping
        TODO: Could set area based on threshold distance (optimal car stopping distance)
        """
        cnts, hierarchy = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        primary_regions = []
        for cnt in cnts:
            if cv2.contourArea(cnt) > min_area:
                epsilon = 0.025*cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                # calculate shape center
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # primary_cnts.append((len(approx), (cX, cY)))
                # draw boudnign rectangle if higher than area
                x,y,w,h = cv2.boundingRect(cnt)
                # print(abs())
                if(abs(w/h - 1) < self.squareness_threshold):
                    # frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    primary_regions.append((x,y,w,h))
                # cv2.drawContours(frame,[approx],0,(0,255,0),3)
        # print("number of vertices in shape: ", len(approx))
        return primary_regions



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
    def feature_mapper(self,new_region,old_region):
        """
        :param new_region: cv_image of new region
        :param old_region: cv_image of old region
        :return: matched keypoints between regions via ORB
        """
        old_region_gray = cv2.cvtColor(old_region,cv2.COLOR_BGR2GRAY)
        new_region_gray= cv2.cvtColor(new_region, cv2.COLOR_BGR2GRAY)
        cv2.imshow('old_region', old_region)
        cv2.imshow('new_region', new_region)
        kp_old,des_old = self.orb.detectAndCompute(old_region_gray,None)
        kp_new,des_new = self.orb.detectAndCompute(new_region_gray,None)
        matches = self.bf.match(des_old, des_new)
        print("matches found between old and new: ", len(matches))
        # store all the good matches as per Lowe's ratio test.
        # good_matches = []
        # for m,n in matches:
        #   if m.distance < 0.7*n.distance:
        #         good_matches.append(m)
        matches = sorted(matches, key = lambda x:x.distance)
        return len(matches)

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
    # det.main()
    det.main_video_processor()

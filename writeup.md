presence# Writeup

## Goal

The goal of our project was to implement a robot controller capable of responding to street signs in the near vicinity by detecting the sign and passing on that information to a robot, for the robot to react to the sign based on its type. We ran into a couple unforeseen roadbumps along the way, and ended with a final MVP of a project capable of recognizing a stop sign and stopping the motion of a car upon approaching the stop sign at a certain distance.

## Solution Overview
Our solution was implemented as two separate ROS nodes, with one in charge of controlling robot behavior, and one in charge of handling sign detection.
We implemented our code with the OSRF Car Demo Prius Simulator, which granted us a lot of flexibility in how we control our robot and read it's sensor data. The simulation featured a Prius robot model with 40 sensors available (each published to a topic) and the ability to control the Prius via continuous throttle, brake, and steering.

The sign detection rosnode was called on every image from the Prius front camera, and operated using color filtering and feature mapping to detect signs. It published its output as a SignMatch message to the `/sign_matched` topic.

The robot controller node subscribed to the `/sign_matched` topic and altered its behavior based on the presence of detected signs. It's default behavior was to drive forwards in a straight line continuously.  

### Color Filtering

To get a good outline of the red of the stop and yield sign, and the yellow of the road sign, we used color filtering. First we converted the image into HSV (Hue, Saturation, Value) to detect the red and yellow parts of the sign easier than RGB. Red would have a hue between 170 and 10 (hue is a circle from 0 to 179, with 0 and 179 being connected), and yellow a hue centered around 24. Both of them have high saturation values, greater than 150. Since value is light based, the signs can have any number of values, so we ignore value. Utilizing all of these into a threshold, we can create a binary image showing just the sign and some small noise.

Using the contour function, we can find contours of the white in the binary image. From this list of contours, we filter them depending on their area. First, we make sure the area is greater than a threshold. We also check the 'squareness' of the sign, to make sure that the sign is not a odd rectangle. If it passes both of these criteria, we deem it a sign.

![](images/hsv.png)

### Feature Matching

We wanted to make sure that the stop sign we are getting in subsequent frames is a consistent result, and not just a momentary glitch. To validate this, we perform feature mapping on the bounding boxes of the potential signs.

We store the box of the previous stop sign in a separate variable to compare to the next frame. Since the bounding box is almost exactly around the stop sign, we add an additional border around the sign so that the feature mapper can pick up more key points of the outside of the sign. We use ORB (Oriented FAST and rotated BRIEF) to collect key points and descriptors in each image. We then run those descriptors through the Brute Force Matcher to get a list of which descriptors are matches. Because the resolution and size of the sign has a big effect on the number of key points and matches these methods return. So we can use this both as a validation and a threshold for when the stop sign is in close proximity to the car.

![](images/compare_regions.png)

### ROS message

To communicate between the sign detection node and the Prius controller node, we created our own custom ROS message named "SignMatch" that contains an int of the number of matches as well as an int that contains the area.

### Prius Control

For the Prius control, we start be getting a message from the sign detection node that contains the area of the sign and the number of matches. Based on this, we decide what to do with the Prius. If the number of matches is zero and the area is zero, we haven't seen anything that looks like a sign, so we set the throttle to 0.1 (which is relatively low acceleration that would be used for maintaining velocity). If the number of matches is zero and the area is a low number, we release the throttle and release the brake. This causes the car to coast, like most people would do if the sign is visible in the distance. If the number of matches is zero and the area is a high number, we release the throttle and gently apply the break because it means the sign is close but not close enough to stop completely. If the number of matches is nonzero, that means we are very close to the stop sign, so we brake hard to stop the car.

## Design decision

## Challenges
-

## Improvements - what would we change if given a second chance
- making image processing less computationally intensive to reduce lag
  - handle edge cases better and avoid calling intensive functions
- improving the quality of feature matching and detection
  - experiment with alternate feature mapping functions
- spending more time with the simulator to better understand the vehicle dynamics
- add more flexibility & scalability & flesh out sign recognition
  - within sign and region tracking over time - noting how many regions are ID'd, which overlap with each other, etc.
  - within recognizing and publishing data about more than 1 sign in the frame
  - within characterizing estimates at sign distance away from the vehicle
  - within characterizing confidence scores of the signs
  - within adding in the ability to recognize more than red signs
- add more to Prius drive control
  - within writing code to recognize Prius motion from sensor data
  - within enabling more complex reaction procedures that factor in car's motion
  - within prioritizing Prius reaction to multiple signs
  - within enabling a constant speed instead of acceleration for the Prius

## Interesting lessons learned
- How to create and call a custom ROS message class to simplify data transfer between nodes
- Writing rosnodes from scratch
- Interacting with and setting up a complex robot simulation environment.
-


# writeup description

* How did you solve the problem (i.e., what methods / algorithms did you use and how do they work)? As above, since not everyone will be familiar with the algorithms you have chosen, you will need to spend some time explaining what you did and how everything works.

* Describe a design decision you had to make when working on your project and what you ultimately did (and why)? These design decisions could be particular choices for how you implemented some part of an algorithm or perhaps a decision regarding which of two external packages to use in your project.

* What if any challenges did you face along the way?

* What would you do to improve your project if you had more time?

* Did you learn any interesting lessons for future robotic programming projects? These could relate to working on robotics projects in teams, working on more open-ended (and longer term) problems, or any other relevant topic.

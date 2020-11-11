# Writeup


* What was the goal of your project? Since everyone is doing a different project, you will have to spend some time setting this context.

## Goal

The goal of our project was to create a robot controller that would respond to street signs by detecting the sign and enabling the robot to react to the sign based on its type. We ran into a couple unforeseen roadbumps along the way and ended with a goal of detecting a stop sign and stopping the car when the stop sign is close enough.

## Solution

## What would we change / add in
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
- add more to prius drive control
  - within writing code to recognize prius motion from sensor data
  - within enabling more complex reaction procedures that factor in car's motion
  - within prioritizing prius reaction to multiple signs
  - within enabling a constant speed instead of acceleration for the prius


### Color Filtering

To get a good outline of the red of the stop and yield sign, and the yellow of the road sign, we used color filtering. First we converted the image into HSV (Hue, Saturation, Value) to detect the red and yellow parts of the sign easier than RGB. Red would have a hue between 170 and 10 (hue is a circle from 0 to 179, with 0 and 179 being connected), and yellow a hue centered around 24. Both of them have high saturation values, greater than 150. Since value is light based, the signs can have any number of values, so we ignore value. Utilizing all of these into a threshold, we can create a binary image showing just the sign and some small noise.

Using the contour function, we can find contours of the white in the binary image. From this list of contours, we filter them depending on their area. First, we make sure the area is greater than a threshold. We also check the 'squareness' of the sign, to make sure that the sign is not a odd rectangle. If it passes both of these criteria, we deem it a sign.

### Feature Matching

We wanted to make sure that the stop sign we are getting in subsequent frames is a consistent result, and not just a momentary glitch. To validate this, we perform feature mapping on the bounding boxes of the potential signs.

Sometimes the sign is

### Prius Control



* How did you solve the problem (i.e., what methods / algorithms did you use and how do they work)? As above, since not everyone will be familiar with the algorithms you have chosen, you will need to spend some time explaining what you did and how everything works.

* Describe a design decision you had to make when working on your project and what you ultimately did (and why)? These design decisions could be particular choices for how you implemented some part of an algorithm or perhaps a decision regarding which of two external packages to use in your project.

* What if any challenges did you face along the way?

* What would you do to improve your project if you had more time?

* Did you learn any interesting lessons for future robotic programming projects? These could relate to working on robotics projects in teams, working on more open-ended (and longer term) problems, or any other relevant topic.

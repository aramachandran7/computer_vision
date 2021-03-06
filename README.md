# computer_vision

## Implementation Plan

* Adi and Nathan E.

* Main idea - Detecting and having our robot react to various different signs and objects and their positions in the surrounding environment. Preferably, using the OSRF's prius simulator.

* Learning Goals - Get more comfortable building ROS codebases from the ground up, and get more comfortable understanding & using computer vision algorithms in our code.

* Algorithms and Computer Vision Areas - Canny Edge Detection, Contours, Machine Learning, Sign Classification

* At first, use the built-in functions, and then choose which ones to implement later based on time and interest. We don't know for sure how easy or hard this will be, and want to leave room for more understanding.

* MVP - Use classical computer vision algorithms (prewritten functions from openCV) with a simulated Neato to detect signs in the Neato's vicinity

* Stretch goal - Use classical CV code (both prewritten and hand-implemented) and perhaps ML for object detection and tracking with the Prius simulator.  

* Risks - time commitment and syncing up when we're both available.

* Need from teaching team - Perhaps getting the Prius simulator set up, and general help as needed.


## Sign detection considerations

packaging - package our function with an input of a still image frame and output of whether or not the sign exists, its type, and how far away from the camera it is.

Use some combination of shape detection and color filtering to pick out individual signs

Write test scripts to benchmark and test the our function's performance in a small datasets

Mess around with confidence scoring

### possible datasets
https://git-disl.github.io/GTDLBench/datasets/lisa_traffic_sign_dataset/

# Reflection

The project code in main.cpp have been extensively documented. For evaluation and learning purposes I will also go through the project creation process and how it works in this document. 

This module covered 3 aspects of self driving car: 
- Prediction
- Trajectory Generation
- Behavior Planning

The applied model is focused on trajectory generation and some part of behavior planing.

I've started this project about a month ago, where little material was released and classes were disconnected (exercises were in python), so I had a hard time putting the basic pieces together. Once the walkthrough was released, the basic pieces got really easy to complete, but the time I had to spend on implementing a proper behavior planning (with a cost function, for example) had already been used. Hence, the implementation attached is simple, and based on what was discussed in the walkthrough with a few modifications to increase performance.

### Trajectory

For the trajectory we will use fernet coordinates, s and d. To generate trajectory, we first pick 5 points far apart. First two points are from the previous trajectory, to ensure continuity (if there is only one point, we infer the previous one using raw), and the next 3 points are each spaced 30 meters ahead in s. The d value will be equal the center of the lane the car will be at.

We use spline interpolation between these 5 base points in order to calculate the remaining points. Using a spline interpolation to generate the points in-between ensures our trajectory is smooth and adheres to the jerk limit set in the simulator. 

The number of points is calculated based on the expected speed and acceleration, since the distance between them defines the acceleration of the ego vehicle.

The calculations are done on local coordinate (from the car's point of view), and then converted back to map coordinate before passing to the simulator.

### Behavior Planning

The implementation of behavior planning is straightforward. We first loop through all vehicles identified in sensor fusion, and update two vectors, lane_speed and lane_pos. These vectors will store the speed and s value of the closest car occupying each lane. So in position 0 of lane_speed vector, we will find the speed of the closest car occupying the left most lane.

This data is useful to determine how our ego vehicle will behave. If there is a car in front of ego, and is going slower, we will prepare to change lanes. Lane changes will respect the road edges, so if ego is at the left most lane, it can only change to the right lane. If ego is at one of the middle lanes, it can change to either the left lane or right lane, with priority given to the left lane (at least where I come from the law says you should change to the left lane to pass a car).

Setting the car to the prep_change lane state doesn't necessarily means it will change lane. It will first check if there is car in the target lane within a safety margin of s (still using frenet coordinates). So if the car is in the middle, wants to change to the left lane but there is car within this safety margin, it will try changing to the right lane instead. If it can not either then it will enter state keep_lane.

At keep_lane state, we will set a avoid collision speed (avc_speed) to the speed of the car in front of the ego vehicle. Ego will slow down to this avc speed until it is safe to change lanes again, in which case it will go for a prep_change lane state, and reset avc_speed to the max speed.

To avoid breaking the acceleration limit, the car speed moves towards its max speed in incremental steps, respecting the max acceleration.

### Future Work

A lot of work can be done to improve this project. The most obvious one would be to generate several candidate trajectories, and implement a cost function to decide which trajectory is best for the ego vehicle. 

In order to build the cost function, we also need to implement a prediction model in order to foresee the future state of the other vehicles identified in the sensor fusion module.

Finally, behavior planning can also be improved by increasing the number of possible behaviors. 














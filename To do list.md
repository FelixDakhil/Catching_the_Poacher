To do list

- Finding out which local navigation algorithm to use
- Figure out how the scan topic works
- Implement said scan topic
- Figure out the main coordinate map
- Find out how to make the robot path towards a waypoint (includes global navigation)


2 
- Find the visualization of the points
- Maybe this could world with the 2d goal pose


Presentation:

Define the problem:
- Drone finds poacher
- Poacher moves
- Drone is fixed wing
- Drone has to follow the poacher

List assumptions:
- Drone has a variable forward velocity, but should be kept the same
- Drone doesnt have to interact with the poacher, ergo 2D can be assumed (we also assume objects dont get in way)
- The environment changes too much for the straight mapping out to be useful
- The poacher may move when the drone sees it, but wont counteract it (i.e. shoot it)
- Poacher moves slower than the drone maximum speed
- The drone has a camera or other detection software to model its environment - Something which approximates lidar scans
- The drone has software to acutate the drone correctly and efficiently, accounting for wind and air density changes
- Gazebo environments can be used as a reasonable approximation of the drone environment

Solution Outline:
- The main goal is to develop the algorithm which steers the drones path, not necessarily the acutal code which accounts for the kinematics and winds. 
- Subsumption Architechture coupled with See-Plan-Act
- Note that the Subsumption architecture is purely due to the visual status of the Poacher, and the main pathing is still all see plan act
- There are three states: 
    1) Looking for a poacher when not visible (includes lookign for him the first time)
    2) Following the poacher when visible
    3) Circling when too near to the poacher (since it is fixed wing)
- Using a see-plan-act local planner with the global planner referring to the drone and poacher location
- Note that the global planner is not the act of giving the location of the poacher, but works towards the poacher (its real not coords every so often)
- The poacher will be modeled by a drone running a slower, worse pathing code which when seen leaks coordinates -- which are used to make up the global pathing code
- Make a little graphs which detail the finished architecture of the drone and current one

Design choices (show an image of the Gazebo Workspace)
1) Drone The current bot being used is the turtlebot 3 since it has the two wheels in front and only goes forward
    - It is chosen because its movement has more similarities with any other code I encountered
    - Later on I can change the visual model with some fixed wing drone if it makes it look better
2) Current Working Environment:
    - Still subject to change, but the given terrain generator doesnt work with the turtlebot due to a change in height, one we do not account for
    - The round objects in the current environment mimic trees and the focus right now is still on the proof of concept
    - Note that the current environment will not be the final one, it being bounded on every side can mess with the unmasked Polar Histogram
3) VFH+ algorithm was used since it is the best out of its competitors for a fixed wing dron
    - The drone requires an algorithm which doesnt have to slow down
    - The area is very large and can change, so cost maps dont work on the local level 
    - This excludes Grid-based (A*, Dijkstra, D* Lite, Jump Point Search, and more)
    - Sampling-based algorithms (RRT, RRT*, PRM) build a tree of collision-free paths and then choose the shortest one
    - These dont work since we often dont have the time to build a really nice tree of paths and then we might find a mid one
    - This leaves us with Reactive algorithms, but DWA samples different velocity commands, which doesnt really work due to the drone not being able to change its speed (NOT velocity) rapidly
    - VFH+ immediately paths towards the local minima of the Polar Histogram of obstacle density, this makes it as responsive as possible

Work I have done:
- The robot works, it can move and doesnt bump into anything
- This is the second iteration of three VFH+ codes
- I sourced the first one, then I tweaked it alot
- The current one has to tragically include an emergency break when the robot gets too close to the pillars
- The robot is also currently stuck trying to path towards the middle of each gap between pillars, due to the surrounding walls
- The 3rd iteration of the VFH+ algorithm was a bounded version, which does not work with the robot since when a region passes the boundary, the robot would have to turn immediately 
- So, we are back to using the second version, if its broke, dont fix it
- Also the algorithm works with points placed while it runs, so a global algorithm implmenetation is manageble 
- The code works in 3 nodes:
1) The see_node takes the single lidar points published by /scan and establishes the obstacle density histogram on /processed_scan and the image on /scan_metadata
2) The plan_node artifically increases the size of the robot (to have a safety margin) and then plans the orientation towards the lowest point, outputting the heading as /vfh_command to the act node (lin.vel, ang.vel)
3) The act node converts the /vfh_command to robot-understadable paramters and timestamps it. This results in a TwistStamped format, which is published over /cmd_vel to the gazebo plant
4) The /scan_metadata is taken and parsed through an lidar_dashboard.html file, which then displays the histogram on the browser for troubleshooting
- Not that this doesnt use NAV2 since the design still requries

Key preformance indicators:
Note: these are used to evaluate the finished product compared to other version of the code I have. I could change the short pathing algorithm (VFH+) or the long searching algorithms like when the drone is unknown.
- Distance decrease over time
- Fastest drone velocity that could be handeled
- Average amount of time spent to find 
- Fastest velocity it can follow 
- If it can adapt to doubling back 

Future work: 
- Obviously implementing the global navigation
- Simulating the runner (but this should take alot less time)
- Changing the work environment
- Working on Key performance indicators (KPIs) to back my findings
- Searching algorithm
- Ask the other people, what would they like to see the most, the single best algorithm, or the pros and cons of multiple
- Ask for any questions and or points of improvement


Global planning:
- We assume that the air density and winds can be ignored which allows us to use a uniform-cost search algorithm, D* Lite

more do to:
- 

- make sure the local planner works (done)
- make a single run file for the local planner (done)
- implement the global planner 
- upload this to a github 
- start on report
- look over code and make sure it works
- Look into ray casting
- Look into how I want to use the point
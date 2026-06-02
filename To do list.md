add , commit -m , push

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
- Drone doesnt have to interact with the poacher, ergo 2D can be assumed (we also assume objects dont get in the way)
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
- We assuthat the air density and winds can be ignored which allows us to use a uniform-cost search algorithm, D* Lite
- we currently assume that all sensors are exact, so that no uncertainty needs to be accounted for (i.e. no Kalman filters or data sampling required)

Questions:
- Ask if I should include Simultaneous Mapping and Localization (SLAM) (dont thinme k so since everything is perfectly accurate)
- Should I include talking about the acutal code used, or the code stolen or just mention it in passing
- 

Issues to write about:
- Local planner:
    - VFH+ oscillation (SOLVED)
    - New point translation (SOLVED)
- Global planner: 
    - Mesh prediction (SOLVED)

- make sure the local planner works (done)
- make a single run file for the local planner (done)
- implement the global planner (done)
- upload this to a github (Done)
- start on report (Done)
- look over code and make sure it works (done)
- Look into ray casting (moot)
- Look into how I want to use the point ()



Report structure:
1. Introduction (2-4)
    - Explain the necessity for poacher detection
    - Explain what the robotics department here is trying to do exactly with the drone
    - Show the actual drone reference
    - Give a model of the area, meaning the evironment of the drone and main problem
    - Define the problem, list assumptions and requirments
2. Robotics Background (5)
    - Explain the node system
    - Explain subsumption and SPA achitectures
    - Explain the idea of a world map and memory
    - 
3. Optimal Search Theory (2-4)
    - Explain that this is NOT economics search theory
    - Based on Koopmans Search and Screening
    - Explain the probability map and whatnot
4. Robotics Implementation (4-8)
    - Local Planner implementation
    - Why VFH+
    - Issues with oscillation and new point translation
    - Adjustments for a fixed wing drone
    - Global Planner implementation
    - Cost Map implementation and world map
    - D*Lite and why it was chosen
5. Search theory implementation and goal (3-5)
    - 
6. Goal evaluation (2 - 3)
    - Looking at KPIs like time to find and whatnot

7. Conclusion (2)
    - What could be added: Adversarial game theory 
    - 

Deadlines: 
    - Presentation 29th
    - Report around the 27th 


Finished Assumptions
    - Environment has obstacles the drone must navigate around
    - Environment has negiligible elevation / Drone can compensate
    - Environment is fenced in, when the fence is broken, the drone is alerted (means we know the starting location of the poacher)
    - Environment obstacles are either there or non-existent, no viewblocks n shit
    - Environment is NOT stationary, drone cannot just run on a world map
    - End result = environment can be approximated by a Gazebo evironment
    -
    - Drone is fixed wing
    - Drone has min velocity
    - Drone cannot go backwards
    - Drone has a turning radius 
    - Drone is equipped with sensory suite
    - Drone is faster thn its own)
    - Drone can be approximated in 2d
    - an the poacher
    - Drone drone doesnt have to physically interact with the poacher
    - Drone can maneuver on its own (I just have to tell it the direction, it can acutate o
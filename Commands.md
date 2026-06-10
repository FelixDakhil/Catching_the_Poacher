Troubleshooting:
    1. Shutting down ROS2
pkill -9 -f gz
pkill -9 -f gazebo
pkill -9 -f ruby
pkill -9 ros2



General Commands

    1. Directory
cd Desktop/ROS2/TB3_WS/CE

    2. Initalization
source ~/turtlebot3_ws/install/setup.bash

    3. Simulation Start
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

    4. Detection
python3 poacher_detection_node.py

    5. Mission Node
python3 mission_node.py



Fixed-Wing-Drone Commands

    1. Local Planner
python3 Local_Planner/quick_start.py

    2. Global Planner
python3 Global_Planner/global_planner_node.py
python3 Global_Planner/cost_map.py

    3. Waypoints
ros2 topic pub --once /global_goal geometry_msgs/msg/Point "{x: 0.0, y: 0.0, z: 0.0}"



Poacher Drone

    1. Spawn Poacher Drone
ros2 launch ~/Desktop/ROS2/TB3_WS/CE/Poacher/poacher_launch.py x_pose:=2.0 y_pose:=0.0

    2. Poacher Drone relay / Waypoints
python3 Poacher/poacher_node.py --ros-args   -p waypoints:="0.5,0.0; 3.0, 2.0;"

    3. Poacher Manual Control
    Window 1:
ros2 launch ~/Desktop/ROS2/TB3_WS/CE/Poacher/poacher_teleop_launch.py
    Window 2:
ros2 run teleop_twist_keyboard teleop_twist_keyboard   --ros-args --remap /cmd_vel:=/poacher/cmd_vel




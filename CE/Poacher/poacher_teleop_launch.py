#!/usr/bin/env python3
"""
poacher_teleop_launch.py  –  Alternative to poacher_node.py.

Starts the poacher infrastructure (spawn + bridge + TF relay) exactly like
poacher_launch.py, but instead of Nav2 it connects a keyboard teleop to
/poacher/cmd_vel so you can drive the poacher manually.

Run AFTER turtlebot3_world.launch.py:
  ros2 launch /path/to/Poacher/poacher_teleop_launch.py x_pose:=2.0 y_pose:=0.0

Controls (teleop_twist_keyboard):
  i / , – forward / backward
  j / l – rotate left / right
  k     – stop
  q / z – increase / decrease speed

The keyboard window must have focus for commands to register.

Note – speed cap
-----------------
teleop_twist_keyboard's own 'speed' parameter only sets the STARTING
linear speed; pressing 'q' still lets the operator increase it past any
configured value, so this is a soft starting point, not a hard ceiling.
POACHER_MAX_SPEED below is intended to match the Nav2 cap in
nav2_poacher_params.yaml (FollowPath.max_vel_x / velocity_smoother
max_velocity[0]) and the poacher_max_speed passed to kpi_recorder_node.py,
so change it in all three places together between iterations.
"""

import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE        = os.path.dirname(os.path.abspath(__file__))
NS          = 'poacher'
TB3_SHARE   = get_package_share_directory('turtlebot3_gazebo')
MODEL       = os.environ.get('TURTLEBOT3_MODEL', 'burger')
SDF_FILE    = os.path.join(HERE, 'turtlebot3_burger_poacher.sdf')
BRIDGE_YAML = os.path.join(HERE, 'turtlebot3_burger_poacher_bridge.yaml')
URDF_FILE   = os.path.join(TB3_SHARE, 'urdf', f'turtlebot3_{MODEL}.urdf')
PYTHON      = sys.executable

POACHER_X = '2.0'
POACHER_Y = '0.0'
DRONE_X   = '-2.0'
DRONE_Y   = '-0.5'

# ── POACHER SPEED CAP – change this between iterations ──────────────────────
# Sets teleop_twist_keyboard's STARTING linear speed (it's a soft cap – see
# the module docstring above). Keep in sync with nav2_poacher_params.yaml
# and kpi_recorder_node.py's poacher_max_speed parameter.
POACHER_MAX_SPEED = 0.18   # m/s
POACHER_MAX_TURN  = 1.0    # rad/s – unchanged from teleop_twist_keyboard default

with open(URDF_FILE, 'r') as f:
    ROBOT_DESC = f.read()


def generate_launch_description():

    x_pose = LaunchConfiguration('x_pose', default=POACHER_X)
    y_pose = LaunchConfiguration('y_pose', default=POACHER_Y)

    # Remove old poacher if present
    remove_old = ExecuteProcess(
        cmd=[
            'gz', 'service',
            '-s', '/world/default/remove',
            '--reqtype', 'gz.msgs.Entity',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', 'name: "poacher" type: MODEL',
        ],
        output='screen',
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', NS, '-file', SDF_FILE,
                   '-x', x_pose, '-y', y_pose, '-z', '0.01'],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='poacher_gz_bridge',
        arguments=['--ros-args', '-p', f'config_file:={BRIDGE_YAML}'],
        output='screen',
    )

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=NS,
        parameters=[{
            'use_sim_time': True,
            'robot_description': ROBOT_DESC,
            'frame_prefix': f'{NS}/',
        }],
        remappings=[('/tf', '/tf'), ('/tf_static', '/tf_static')],
        output='screen',
    )

    tf_relay = ExecuteProcess(
        cmd=[PYTHON, os.path.join(HERE, 'poacher_tf_relay.py'),
             '--ros-args',
             '-p', f'poacher_x:={POACHER_X}',
             '-p', f'poacher_y:={POACHER_Y}',
             '-p', f'drone_x:={DRONE_X}',
             '-p', f'drone_y:={DRONE_Y}',
             ],
        output='screen',
    )

    # Teleop: publishes Twist on /cmd_vel, we remap to /poacher/cmd_vel
    teleop = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='poacher_teleop',
        parameters=[{
            'speed': POACHER_MAX_SPEED,
            'turn':  POACHER_MAX_TURN,
        }],
        remappings=[('/cmd_vel', '/poacher/cmd_vel')],
        output='screen',
        prefix='xterm -e',   # opens in its own window so keyboard focus works
    )

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('x_pose', default_value=POACHER_X))
    ld.add_action(DeclareLaunchArgument('y_pose', default_value=POACHER_Y))
    ld.add_action(remove_old)
    ld.add_action(TimerAction(period=1.5, actions=[spawn]))
    ld.add_action(TimerAction(period=3.0, actions=[bridge, rsp, tf_relay]))
    ld.add_action(TimerAction(period=4.0, actions=[teleop]))
    return ld

#!/usr/bin/env python3
"""
poacher_launch.py  –  Complete poacher stack, clean start.

Files needed in the same folder:
  poacher_launch.py
  turtlebot3_burger_poacher.sdf
  turtlebot3_burger_poacher_bridge.yaml
  nav2_poacher_params.yaml
  poacher_tf_relay.py
  poacher_node.py

Run AFTER turtlebot3_world.launch.py:
  ros2 launch /path/to/Poacher/poacher_launch.py x_pose:=2.0 y_pose:=0.0
"""

import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, GroupAction, TimerAction
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushROSNamespace

HERE        = os.path.dirname(os.path.abspath(__file__))
NS          = 'poacher'
TB3_SHARE   = get_package_share_directory('turtlebot3_gazebo')
MODEL       = os.environ.get('TURTLEBOT3_MODEL', 'burger')
# Use custom SDF with model name="poacher" so gz topics are /poacher/*
SDF_FILE    = os.path.join(HERE, 'turtlebot3_burger_poacher.sdf')
BRIDGE_YAML = os.path.join(HERE, 'turtlebot3_burger_poacher_bridge.yaml')
NAV2_PARAMS = os.path.join(HERE, 'nav2_poacher_params.yaml')
URDF_FILE   = os.path.join(TB3_SHARE, 'urdf', f'turtlebot3_{MODEL}.urdf')
PYTHON      = sys.executable

with open(URDF_FILE, 'r') as f:
    ROBOT_DESC = f.read()


def generate_launch_description():

    x_pose = LaunchConfiguration('x_pose', default='2.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    # ------------------------------------------------------------------
    # 1. Remove old poacher model from Gazebo if it exists
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Spawn fresh poacher model
    # ------------------------------------------------------------------
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', NS,
            '-file', SDF_FILE,
            '-x', x_pose,
            '-y', y_pose,
            '-z', '0.01',
        ],
        output='screen',
    )

    # ------------------------------------------------------------------
    # 3. Bridge gz /poacher/* → ROS /poacher/*
    # ------------------------------------------------------------------
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='poacher_gz_bridge',
        arguments=['--ros-args', '-p', f'config_file:={BRIDGE_YAML}'],
        output='screen',
    )

    # ------------------------------------------------------------------
    # 4. TF relay: rewrites plain frame names → poacher/* frame names
    #    Runs as a plain Python script so no package install needed
    # ------------------------------------------------------------------
    tf_relay = ExecuteProcess(
        cmd=[PYTHON, os.path.join(HERE, 'poacher_tf_relay.py'),
             '--ros-args',
             '-p', 'poacher_x:=2.0',
             '-p', 'poacher_y:=0.0',
             '-p', 'drone_x:=-2.0',
             '-p', 'drone_y:=-0.5',
             ],
        output='screen',
    )

    # ------------------------------------------------------------------
    # 5. robot_state_publisher for /poacher/tf_static (URDF joints)
    # ------------------------------------------------------------------
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
        remappings=[
            ('/tf',        '/tf'),
            ('/tf_static', '/tf_static'),
        ],
        output='screen',
    )

    # ------------------------------------------------------------------
    # 6. Nav2 nodes under /poacher namespace
    # ------------------------------------------------------------------
    nav2_group = GroupAction([
        PushROSNamespace(NS),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            parameters=[NAV2_PARAMS],
            remappings=[
                ('cmd_vel', f'/{NS}/cmd_vel'),
                ('odom',    f'/{NS}/odom'),
            ],
            output='screen',
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            parameters=[NAV2_PARAMS],
            output='screen',
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            parameters=[NAV2_PARAMS],
            remappings=[('cmd_vel', f'/{NS}/cmd_vel')],
            output='screen',
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            parameters=[NAV2_PARAMS],
            output='screen',
        ),
        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            parameters=[NAV2_PARAMS],
            output='screen',
        ),
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            parameters=[NAV2_PARAMS],
            remappings=[
                ('cmd_vel',          f'/{NS}/cmd_vel'),
                ('cmd_vel_smoothed', f'/{NS}/cmd_vel'),
            ],
            output='screen',
        ),
    ])

    # ------------------------------------------------------------------
    # 7. Lifecycle manager – inside namespace, short node names
    # ------------------------------------------------------------------
    lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_poacher',
        namespace=NS,
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'bond_timeout': 0.0,
            'node_names': [
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'waypoint_follower',
                'velocity_smoother',
            ],
        }],
        output='screen',
    )

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('x_pose', default_value='2.0'))
    ld.add_action(DeclareLaunchArgument('y_pose', default_value='0.0'))
    ld.add_action(remove_old)
    ld.add_action(TimerAction(period=1.5, actions=[spawn]))
    ld.add_action(TimerAction(period=3.0, actions=[bridge, rsp, tf_relay]))
    ld.add_action(TimerAction(period=4.0, actions=[nav2_group]))
    ld.add_action(TimerAction(period=12.0, actions=[lifecycle]))
    return ld
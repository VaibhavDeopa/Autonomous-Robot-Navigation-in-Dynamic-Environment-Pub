#!/usr/bin/env python3
"""
navigation_sim.launch.py — ROS2 Launch file for DRL Navigation Training
========================================================================
Launches:
  1. Webots simulation with the TurtleBot3 arena
  2. TurtleBot3 ROS2 driver (publishes /scan, /odom, subscribes to /cmd_vel)
  3. Supervisor controller (provides /reset_robot service)

Usage:
  ros2 launch nav_env_pkg navigation_sim.launch.py
"""

import os
import pathlib

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

# Try to import webots_ros2 launch utilities
try:
    from webots_ros2_driver.webots_launcher import WebotsLauncher
    from webots_ros2_driver.webots_controller import WebotsController
    WEBOTS_ROS2_AVAILABLE = True
except ImportError:
    WEBOTS_ROS2_AVAILABLE = False


def generate_launch_description():
    package_dir = get_package_share_directory('nav_env_pkg')

    # Path to the world file
    world_file = os.path.join(package_dir, 'worlds', 'turtlebot3_arena.wbt')

    # ── Webots Simulator ──────────────────────────────────────────────
    webots = WebotsLauncher(
        world=world_file,
        mode='realtime',  # Use 'fast' for training speed
    )

    # ── TurtleBot3 ROS2 Driver ────────────────────────────────────────
    # This connects to the TurtleBot3 in Webots and publishes
    # /scan (LaserScan), /odom (Odometry), subscribes to /cmd_vel (Twist)
    turtlebot_driver = WebotsController(
        robot_name='TurtleBot3Burger',
        parameters=[
            {'robot_description': ''},  # Uses default URDF from webots_ros2
        ],
    )

    # ── Supervisor Controller ─────────────────────────────────────────
    # Runs the supervisor_controller.py as a Webots external controller
    supervisor_controller = WebotsController(
        robot_name='supervisor',
        parameters=[],
    )

    return LaunchDescription([
        LogInfo(msg='🚀 Launching DRL Navigation Training Environment'),
        LogInfo(msg=f'   World: {world_file}'),

        # Start Webots
        webots,

        # Start TurtleBot3 driver
        turtlebot_driver,

        # Start Supervisor
        supervisor_controller,

        # Shutdown when Webots closes
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(
                    event=launch.events.Shutdown()
                )],
            )
        ),
    ])

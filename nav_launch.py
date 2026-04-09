#!/usr/bin/env python3
"""
nav_launch.py — Launch Webots + TurtleBot3 + Supervisor for DRL training
Supports both circular (turtlebot3_arena.wbt) and square (square_arena.wbt).
"""
import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController


def generate_launch_description():
    package_dir = get_package_share_directory('nav_env_pkg')

    # ── Select the arena world (square for curriculum learning) ────────
    world = os.path.join(package_dir, 'worlds', 'square_arena.wbt')

    robot_urdf = os.path.join(package_dir, 'resource', 'turtlebot3_burger.urdf')
    supervisor_urdf = os.path.join(package_dir, 'resource', 'supervisor.urdf')
    ros2_control_params = os.path.join(
        package_dir, 'resource', 'ros2control.yml'
    )

    webots = WebotsLauncher(
        world=world,
        mode='fast',
    )

    # ── TurtleBot3 driver (LiDAR + DiffDrive) ────────────────────
    turtlebot_driver = WebotsController(
        robot_name='TurtleBot3Burger',
        parameters=[
            {'robot_description': robot_urdf},
            ros2_control_params,
        ],
    )

    # ── Supervisor driver (reset + dynamic obstacles) ─────────────
    supervisor_driver = WebotsController(
        robot_name='supervisor',
        parameters=[
            {'robot_description': supervisor_urdf},
        ],
    )

    # ── Controller spawners ───────────────────────────────────────
    diffdrive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['diffdrive_controller', '--controller-manager-timeout', '120'],
    )

    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '120'],
    )

    return LaunchDescription([
        webots,
        turtlebot_driver,
        supervisor_driver,
        diffdrive_spawner,
        joint_state_spawner,

        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(
                    event=launch.events.Shutdown()
                )],
            )
        ),
    ])

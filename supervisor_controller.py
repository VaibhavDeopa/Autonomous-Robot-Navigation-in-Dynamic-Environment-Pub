#!/usr/bin/env python3
"""
supervisor_controller.py — Webots Supervisor ROS2 Node
======================================================
Provides services for resetting / teleporting the TurtleBot3 and
controlling dynamic obstacles in the Webots simulation.

Services:
    /reset_robot    (std_srvs/Empty)   → Reset robot to start + reset physics
    /teleport_robot (custom or Empty)  → Move robot to (x, y, θ)
    /reset_world    (std_srvs/Empty)   → Full simulation reset

This script runs as a Webots Supervisor controller.
Launch it by assigning it to a Supervisor-type Robot node in the .wbt file.
"""

import math
import sys
import random

try:
    from controller import Supervisor
except ImportError:
    print("[SupervisorController] ⚠  'controller' module not found. "
          "This script must run as a Webots controller.")
    sys.exit(1)

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from geometry_msgs.msg import Point


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Robot start pose [x, y, rotation_angle(rad)]
ROBOT_START_X = 0.0
ROBOT_START_Y = 0.0
ROBOT_START_THETA = 0.0

# TurtleBot3 DEF name in the .wbt world file
ROBOT_DEF_NAME = "TURTLEBOT3"

# Dynamic obstacle DEF names (match your world file)
OBSTACLE_DEF_NAMES = ["OBS_1", "OBS_2", "OBS_3"]

# Arena bounds for random obstacle placement
ARENA_X_MIN, ARENA_X_MAX = -3.5, 3.5
ARENA_Y_MIN, ARENA_Y_MAX = -3.5, 3.5

# Obstacle movement parameters
OBSTACLE_SPEED = 0.3         # m/s
OBSTACLE_CHANGE_DIR_STEPS = 200  # steps before changing direction


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Obstacle Controller
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicObstacle:
    """Manages a single moving obstacle in the simulation."""

    def __init__(self, wb_node, speed: float, change_interval: int):
        self.node = wb_node
        self.speed = speed
        self.change_interval = change_interval
        self.step_counter = 0

        # Random initial direction
        angle = random.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)

        # Get the translation field
        self.trans_field = self.node.getField("translation")

    def update(self, timestep_s: float):
        """Move the obstacle one step; bounce off arena walls."""
        self.step_counter += 1

        # Periodically randomise direction
        if self.step_counter % self.change_interval == 0:
            angle = random.uniform(0, 2 * math.pi)
            self.vx = self.speed * math.cos(angle)
            self.vy = self.speed * math.sin(angle)

        pos = self.trans_field.getSFVec3f()
        new_x = pos[0] + self.vx * timestep_s
        new_y = pos[1] + self.vy * timestep_s

        # Wall bounce
        if new_x < ARENA_X_MIN or new_x > ARENA_X_MAX:
            self.vx *= -1
            new_x = max(ARENA_X_MIN, min(new_x, ARENA_X_MAX))
        if new_y < ARENA_Y_MIN or new_y > ARENA_Y_MAX:
            self.vy *= -1
            new_y = max(ARENA_Y_MIN, min(new_y, ARENA_Y_MAX))

        self.trans_field.setSFVec3f([new_x, new_y, pos[2]])

    def randomise_position(self):
        """Place obstacle at a random position (avoiding robot start)."""
        while True:
            x = random.uniform(ARENA_X_MIN + 0.5, ARENA_X_MAX - 0.5)
            y = random.uniform(ARENA_Y_MIN + 0.5, ARENA_Y_MAX - 0.5)
            # Ensure not too close to robot start
            if math.hypot(x - ROBOT_START_X, y - ROBOT_START_Y) > 1.0:
                break

        pos = self.trans_field.getSFVec3f()
        self.trans_field.setSFVec3f([x, y, pos[2]])

        # New random direction
        angle = random.uniform(0, 2 * math.pi)
        self.vx = self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)


# ═══════════════════════════════════════════════════════════════════════════════
# Supervisor ROS2 Node
# ═══════════════════════════════════════════════════════════════════════════════

class SupervisorController:
    """
    Combines Webots Supervisor API with ROS2 services.
    Runs in the Webots controller process (not a standard ROS2 node lifecycle).
    """

    def __init__(self):
        # ── Webots Supervisor ─────────────────────────────────────────────
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.timestep_s = self.timestep / 1000.0

        # ── Get robot node ────────────────────────────────────────────────
        self.robot_node = self.supervisor.getFromDef(ROBOT_DEF_NAME)
        if self.robot_node is None:
            print(f"[Supervisor] ❌  Robot DEF '{ROBOT_DEF_NAME}' not found in world!")
            print(f"[Supervisor]     Make sure your TurtleBot3 has DEF name: {ROBOT_DEF_NAME}")
            sys.exit(1)

        self.robot_trans_field = self.robot_node.getField("translation")
        self.robot_rot_field = self.robot_node.getField("rotation")

        # ── Dynamic obstacles ─────────────────────────────────────────────
        self.obstacles: list[DynamicObstacle] = []
        for def_name in OBSTACLE_DEF_NAMES:
            obs_node = self.supervisor.getFromDef(def_name)
            if obs_node is not None:
                self.obstacles.append(
                    DynamicObstacle(obs_node, OBSTACLE_SPEED, OBSTACLE_CHANGE_DIR_STEPS)
                )
                print(f"[Supervisor] ✅  Dynamic obstacle '{def_name}' registered.")
            else:
                print(f"[Supervisor] ⚠  Obstacle DEF '{def_name}' not found — skipping.")

        # ── ROS2 ──────────────────────────────────────────────────────────
        rclpy.init(args=None)
        self.ros_node = Node("webots_supervisor")

        # Services
        self.reset_robot_srv = self.ros_node.create_service(
            Empty, "/reset_robot", self._handle_reset_robot
        )
        self.reset_world_srv = self.ros_node.create_service(
            Empty, "/reset_world", self._handle_reset_world
        )

        # Publisher for goal position (optional, for visualisation)
        # self.goal_pub = self.ros_node.create_publisher(Point, "/goal_position", 10)

        self.ros_node.get_logger().info(
            "🎮  Supervisor controller ready. "
            f"Robot='{ROBOT_DEF_NAME}', "
            f"Obstacles={len(self.obstacles)}"
        )

    # ── Service handlers ──────────────────────────────────────────────────

    def _handle_reset_robot(self, request, response):
        """Reset robot to start position and clear its physics."""
        self.ros_node.get_logger().info("🔄  Resetting robot to start position...")
        self._teleport_robot(ROBOT_START_X, ROBOT_START_Y, ROBOT_START_THETA)

        # Randomise obstacle positions too
        for obs in self.obstacles:
            obs.randomise_position()

        return response

    def _handle_reset_world(self, request, response):
        """Full simulation reset (physics + time)."""
        self.ros_node.get_logger().info("🌍  Full world reset...")
        self.supervisor.simulationReset()
        self.supervisor.step(self.timestep)  # one step to apply

        # Re-teleport robot (simulation reset may restore initial state)
        self._teleport_robot(ROBOT_START_X, ROBOT_START_Y, ROBOT_START_THETA)

        for obs in self.obstacles:
            obs.randomise_position()

        return response

    # ── Robot teleportation ───────────────────────────────────────────────

    def _teleport_robot(self, x: float, y: float, theta: float):
        """Move the robot to (x, y) with heading theta."""
        # Set translation (keep z the same as current)
        current_pos = self.robot_trans_field.getSFVec3f()
        self.robot_trans_field.setSFVec3f([x, y, current_pos[2]])

        # Set rotation — Webots uses axis-angle: [0, 0, 1, angle]
        self.robot_rot_field.setSFRotation([0.0, 0.0, 1.0, theta])

        # Reset physics to clear velocity / forces
        self.robot_node.resetPhysics()

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        """Main simulation loop: step Webots + spin ROS2 + move obstacles."""
        while self.supervisor.step(self.timestep) != -1:
            # Process ROS2 callbacks (non-blocking)
            rclpy.spin_once(self.ros_node, timeout_sec=0.0)

            # Update dynamic obstacles
            for obs in self.obstacles:
                obs.update(self.timestep_s)

    def cleanup(self):
        """Shutdown ROS2."""
        self.ros_node.destroy_node()
        rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point (called by Webots)
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    controller = SupervisorController()
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()

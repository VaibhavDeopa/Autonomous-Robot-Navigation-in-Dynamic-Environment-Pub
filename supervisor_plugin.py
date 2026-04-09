#!/usr/bin/env python3
"""
supervisor_plugin.py — Phase-Aware Supervisor for Curriculum Learning
=====================================================================
Runs as a webots_ros2 plugin inside the launch file.

Reads CURRICULUM_PHASE env var (default=1) to decide which obstacles
are active and how dynamic obstacles move:

  Phase 1: Empty room      — all obstacles hidden under floor (z=-10)
  Phase 2: Static only     — 4 static boxes visible, dynamic hidden
  Phase 3: Slow dynamic    — all obstacles visible, dynamic speed=0.10 m/s
  Phase 4: Fast dynamic    — all obstacles visible, dynamic speed=0.30 m/s

Provides:
  - /reset_robot service   → teleport TurtleBot3 + configure obstacles
  - Dynamic obstacle movement each simulation step (phases 3-4)
"""

import math
import os
import random
import threading
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import Point

# ── Phase configuration ──────────────────────────────────────────────────
PHASE_CONFIG = {
    1: {"static": False, "dynamic": False, "speed": 0.0,  "label": "Empty Room"},
    2: {"static": True,  "dynamic": False, "speed": 0.0,  "label": "Static Obstacles"},
    3: {"static": True,  "dynamic": True,  "speed": 0.10, "label": "Slow Dynamic"},
    4: {"static": True,  "dynamic": True,  "speed": 0.30, "label": "Fast Dynamic"},
}

# Arena bounds (3m × 3m square, walls at ±1.5)
ARENA_HALF = 1.2   # stay 0.3m away from walls
SPAWN_HALF = 1.0   # robot/goal spawn zone (safer inner region)
HIDDEN_Z = -10.0   # teleport obstacles here to "hide" them


def _safe_float(v, default=0.0):
    """Return v if finite, else default. Prevents NaN propagation."""
    return v if math.isfinite(v) else default


def _clamp(val, lo, hi):
    """Clamp val to [lo, hi], returning default if NaN."""
    if not math.isfinite(val):
        return (lo + hi) / 2.0
    return max(lo, min(hi, val))


class SupervisorPlugin:
    """
    webots_ros2_driver plugin that controls the simulation.
    Called by the driver at each Webots timestep.
    """

    def init(self, webots_node, properties):
        """Called once when the plugin is loaded."""
        try:
            self._robot = webots_node.robot   # Supervisor instance
            self._node = webots_node          # ROS2 node

            self._timestep = int(self._robot.getBasicTimeStep())

            # ── Read curriculum phase ─────────────────────────────────────
            self._phase = int(os.environ.get("CURRICULUM_PHASE", "1"))
            self._phase = max(1, min(4, self._phase))  # clamp 1-4
            self._cfg = PHASE_CONFIG[self._phase]
            print(f"[SUPERVISOR] Phase = {self._phase}, config = {self._cfg}", flush=True)

            # ── Get references to scene objects ───────────────────────────
            self._turtlebot = self._robot.getFromDef("TURTLEBOT3")
            print(f"[SUPERVISOR] TurtleBot found: {self._turtlebot is not None}", flush=True)

            # ── Visual markers (green=goal, blue=start) ──────────────────
            self._goal_marker = self._robot.getFromDef("GOAL_MARKER")
            self._start_marker = self._robot.getFromDef("START_MARKER")
            print(f"[SUPERVISOR] GOAL_MARKER: {self._goal_marker is not None}", flush=True)
            print(f"[SUPERVISOR] START_MARKER: {self._start_marker is not None}", flush=True)

            self._static_obs = []
            for i in range(1, 5):
                obj = self._robot.getFromDef(f"STATIC_{i}")
                print(f"[SUPERVISOR] STATIC_{i}: {obj is not None}", flush=True)
                if obj is not None:
                    self._static_obs.append(obj)

            self._dynamic_obs = []
            for i in range(1, 4):
                obj = self._robot.getFromDef(f"OBS_{i}")
                print(f"[SUPERVISOR] OBS_{i}: {obj is not None}", flush=True)
                if obj is not None:
                    self._dynamic_obs.append(obj)

            # ── Dynamic obstacle state ────────────────────────────────────
            self._obs_velocities = []
            self._obs_speed = self._cfg["speed"]

            for _ in self._dynamic_obs:
                angle = random.uniform(0, 2 * math.pi)
                self._obs_velocities.append([
                    self._obs_speed * math.cos(angle),
                    self._obs_speed * math.sin(angle),
                ])

            # ── Store original static positions (for randomising) ────────
            self._static_home_positions = []
            for obs in self._static_obs:
                pos = obs.getField("translation").getSFVec3f()
                self._static_home_positions.append([pos[0], pos[1], pos[2]])

            # ── Recovery state ────────────────────────────────────────────
            self._in_recovery = False
            self._last_recovery_time = 0.0
            self._recovery_cooldown = 1.0  # seconds between recoveries
            self._consecutive_nan_count = 0

            # ── ROS2 reset service ────────────────────────────────────────
            self._ctx = rclpy.Context()
            self._ctx.init()
            self._service_node = Node(
                'supervisor_reset_service', context=self._ctx
            )
            self._reset_srv = self._service_node.create_service(
                Trigger, "/reset_robot", self._handle_reset
            )

            # ── Subscribe to goal pose from nav_env ───────────────────────
            self._goal_sub = self._service_node.create_subscription(
                Point, '/goal_pose', self._goal_pose_cb, 1
            )

            self._executor = rclpy.executors.SingleThreadedExecutor(
                context=self._ctx
            )
            self._executor.add_node(self._service_node)
            self._spin_thread = threading.Thread(
                target=self._executor.spin,
                daemon=True
            )
            self._spin_thread.start()

            # ── Ground truth position publisher ──────────────────────────
            self._ground_truth_pub = self._service_node.create_publisher(
                Point, '/robot_ground_truth', 1
            )
            print(f"[SUPERVISOR] /reset_robot service + /goal_pose sub + /robot_ground_truth pub created ✅", flush=True)

            # ── Initial obstacle placement based on phase ─────────────────
            self._apply_phase_config()
            print(f"[SUPERVISOR] _apply_phase_config() done for Phase {self._phase}", flush=True)

            self._step_count = 0

            print(f"[SUPERVISOR] ✅ Init complete — Phase {self._phase}: {self._cfg['label']}", flush=True)
            rclpy.logging.get_logger("supervisor_plugin").info(
                f"🎮  Supervisor initialised — Phase {self._phase}: "
                f"{self._cfg['label']}  |  "
                f"Static: {len(self._static_obs)}  Dynamic: {len(self._dynamic_obs)}  "
                f"Speed: {self._obs_speed} m/s"
            )
        except Exception as e:
            print(f"[SUPERVISOR] ❌ INIT FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _apply_phase_config(self):
        """Show/hide obstacles based on current phase. Track positions."""
        self._obstacle_positions = [(0.0, 0.0)]

        # ── Static obstacles ──────────────────────────────────────────
        for i, obs in enumerate(self._static_obs):
            tf = obs.getField("translation")
            if self._cfg["static"]:
                ox, oy = self._random_obstacle_pos(placed=self._obstacle_positions)
                tf.setSFVec3f([ox, oy, 0.16])
                self._obstacle_positions.append((ox, oy))
            else:
                tf.setSFVec3f([0.0, float(i) * 0.5, HIDDEN_Z])

        # ── Dynamic obstacles ─────────────────────────────────────────
        for i, obs in enumerate(self._dynamic_obs):
            tf = obs.getField("translation")
            if self._cfg["dynamic"]:
                ox, oy = self._random_obstacle_pos(placed=self._obstacle_positions)
                tf.setSFVec3f([ox, oy, 0.11])
                self._obstacle_positions.append((ox, oy))
            else:
                tf.setSFVec3f([float(i) * 0.5, 0.0, HIDDEN_Z])

        # ── Publish obstacle positions so nav_env can avoid them ──────
        self._publish_obstacle_positions()

    def _publish_obstacle_positions(self):
        """Publish active obstacle positions on /obstacle_positions."""
        if not hasattr(self, '_service_node'):
            return
        if not hasattr(self, '_obs_pos_pub'):
            from std_msgs.msg import Float32MultiArray
            self._obs_pos_pub = self._service_node.create_publisher(
                Float32MultiArray, '/obstacle_positions', 1
            )
        from std_msgs.msg import Float32MultiArray
        msg = Float32MultiArray()
        flat = []
        for ox, oy in self._obstacle_positions:
            flat.extend([_safe_float(ox), _safe_float(oy)])
        msg.data = flat
        self._obs_pos_pub.publish(msg)

    def _random_obstacle_pos(self, placed: list = None):
        """Random position inside the arena, avoiding placed objects."""
        if placed is None:
            placed = []
        safe_half = ARENA_HALF - 0.2
        min_obs_dist = 0.5

        for _ in range(200):
            ox = random.uniform(-safe_half, safe_half)
            oy = random.uniform(-safe_half, safe_half)

            too_close = False
            for px, py in placed:
                if math.hypot(ox - px, oy - py) < min_obs_dist:
                    too_close = True
                    break
            if too_close:
                continue

            return ox, oy
        return safe_half, safe_half  # fallback

    def _goal_pose_cb(self, msg: Point):
        """Move the green GOAL_MARKER to the published goal position."""
        if self._goal_marker is not None:
            gx = _clamp(msg.x, -ARENA_HALF, ARENA_HALF)
            gy = _clamp(msg.y, -ARENA_HALF, ARENA_HALF)
            self._goal_marker.getField("translation").setSFVec3f(
                [gx, gy, 0.005]
            )

    def _is_safe_from_obstacles(self, x, y, min_dist=0.5):
        """Check if (x, y) is far enough from all active obstacles."""
        for ox, oy in getattr(self, '_obstacle_positions', []):
            if math.hypot(x - ox, y - oy) < min_dist:
                return False
        return True

    def _handle_reset(self, request, response):
        """Teleport robot to random position, reconfigure obstacles."""
        # ── Reconfigure obstacles FIRST ──
        self._apply_phase_config()

        # ── Reset robot position (avoid obstacles) ────────────────────
        rx, ry = 0.0, 0.0
        if self._turtlebot is not None:
            found_safe = False
            for _ in range(200):
                rx = random.uniform(-SPAWN_HALF, SPAWN_HALF)
                ry = random.uniform(-SPAWN_HALF, SPAWN_HALF)
                if self._is_safe_from_obstacles(rx, ry, min_dist=0.4):
                    found_safe = True
                    break

            if not found_safe:
                print("[SUPERVISOR] ⚠ Could not find safe spawn after 200 tries!")
                rx, ry = 0.0, 0.0

            rtheta = random.uniform(-math.pi, math.pi)

            self._turtlebot.getField("translation").setSFVec3f([rx, ry, 0.05])
            self._turtlebot.getField("rotation").setSFRotation([0, 0, 1, rtheta])
            self._robot.simulationResetPhysics()

        # ── Place blue START_MARKER at spawn location ─────────────────
        if self._start_marker is not None:
            self._start_marker.getField("translation").setSFVec3f(
                [rx, ry, 0.005]
            )

        # ── Reset dynamic velocities ──────────────────────────────────
        for i in range(len(self._dynamic_obs)):
            angle = random.uniform(0, 2 * math.pi)
            self._obs_velocities[i] = [
                self._obs_speed * math.cos(angle),
                self._obs_speed * math.sin(angle),
            ]

        # ── Clear recovery state ──────────────────────────────────────
        self._in_recovery = False
        self._consecutive_nan_count = 0

        response.success = True
        response.message = (
            f"Phase {self._phase} reset — robot at ({rx:.2f}, {ry:.2f})"
        )
        return response

    def _recover_robot(self):
        """Lightweight recovery: teleport robot to centre, reset only its physics.
        Does NOT call simulationResetPhysics() which destabilises all objects.
        """
        now = time.monotonic()
        if now - self._last_recovery_time < self._recovery_cooldown:
            return  # cooldown active, skip

        self._last_recovery_time = now
        self._consecutive_nan_count += 1

        # Find a safe spawn point
        rx, ry = 0.0, 0.0
        for _ in range(50):
            rx = random.uniform(-SPAWN_HALF, SPAWN_HALF)
            ry = random.uniform(-SPAWN_HALF, SPAWN_HALF)
            if self._is_safe_from_obstacles(rx, ry, min_dist=0.4):
                break

        rtheta = random.uniform(-math.pi, math.pi)

        # Teleport robot only
        self._turtlebot.getField("translation").setSFVec3f([rx, ry, 0.05])
        self._turtlebot.getField("rotation").setSFRotation([0, 0, 1, rtheta])
        self._robot.simulationResetPhysics()  # Globally reset physics to clear NaN forces

        # Publish safe ground truth immediately
        msg = Point()
        msg.x = float(rx)
        msg.y = float(ry)
        msg.z = float(rtheta)
        self._ground_truth_pub.publish(msg)

        print(f"[SUPERVISOR] ✅ Recovery #{self._consecutive_nan_count}: "
              f"robot → ({rx:.2f}, {ry:.2f})", flush=True)

    def step(self):
        """Called at each Webots simulation step — publish ground truth + move dynamic obstacles."""
        # ── Publish robot's true world position every step ────────────
        if self._turtlebot is not None and hasattr(self, '_ground_truth_pub'):
            try:
                pos = self._turtlebot.getField("translation").getSFVec3f()
                rot = self._turtlebot.getField("rotation").getSFRotation()

                # ── CATASTROPHIC FAILURE WATCHDOG ─────────────────────
                pos_nan = (not math.isfinite(pos[0]) or not math.isfinite(pos[1])
                           or not math.isfinite(pos[2]))
                pos_fallen = (not pos_nan) and (pos[2] < -0.1)
                pos_escaped = (not pos_nan) and (abs(pos[0]) > 2.0 or abs(pos[1]) > 2.0)

                if pos_nan or pos_fallen or pos_escaped:
                    if not self._in_recovery:
                        reason = "NaN" if pos_nan else ("fell" if pos_fallen else "escaped")
                        print(f"[SUPERVISOR] ⚠ PHYSICS FAILURE ({reason}) "
                              f"pos=({_safe_float(pos[0]):.2f}, "
                              f"{_safe_float(pos[1]):.2f}, "
                              f"{_safe_float(pos[2]):.2f}) — recovering",
                              flush=True)
                        self._in_recovery = True
                        self._recover_robot()
                        self._in_recovery = False
                    return

                # ── Reset consecutive NaN counter on good step ────────
                self._consecutive_nan_count = 0

                # ── Extract yaw safely ────────────────────────────────
                # rot = [axis_x, axis_y, axis_z, angle]
                # For Z-axis rotation: yaw = angle * sign(axis_z)
                axis_z = _safe_float(rot[2], 1.0)
                angle = _safe_float(rot[3], 0.0)
                if abs(axis_z) < 0.01:
                    # Robot might be tipped — use angle 0 as fallback
                    yaw = 0.0
                else:
                    yaw = angle * (1.0 if axis_z >= 0 else -1.0)

                # Clamp yaw to [-π, π]
                while yaw > math.pi:
                    yaw -= 2.0 * math.pi
                while yaw < -math.pi:
                    yaw += 2.0 * math.pi

                # ── Publish ground truth ──────────────────────────────
                msg = Point()
                msg.x = _clamp(float(pos[0]), -2.0, 2.0)
                msg.y = _clamp(float(pos[1]), -2.0, 2.0)
                msg.z = float(yaw)  # pack yaw into z field
                self._ground_truth_pub.publish(msg)

            except Exception as e:
                print(f"[SUPERVISOR] step() error: {e}", flush=True)

        # Only move obstacles in phases 3 and 4
        if not self._cfg["dynamic"] or self._obs_speed <= 0:
            return

        dt = self._timestep / 1000.0  # ms → seconds
        self._step_count += 1

        for i, obs in enumerate(self._dynamic_obs):
            pos = obs.getField("translation").getSFVec3f()

            # Skip hidden obstacles
            if pos[2] < -5.0:
                continue

            vx, vy = self._obs_velocities[i]

            # Update position
            new_x = pos[0] + vx * dt
            new_y = pos[1] + vy * dt

            # ── Bounce off square walls ───────────────────────────────
            bounced = False
            if abs(new_x) > ARENA_HALF:
                self._obs_velocities[i][0] = -vx
                new_x = max(-ARENA_HALF, min(ARENA_HALF, new_x))
                bounced = True
            if abs(new_y) > ARENA_HALF:
                self._obs_velocities[i][1] = -vy
                new_y = max(-ARENA_HALF, min(ARENA_HALF, new_y))
                bounced = True

            # ── Random direction change (8% chance per step) ──────────
            if random.random() < 0.08 or bounced:
                angle = random.uniform(0, 2 * math.pi)
                self._obs_velocities[i] = [
                    self._obs_speed * math.cos(angle),
                    self._obs_speed * math.sin(angle),
                ]

            # Clamp position to arena bounds (extra safety)
            new_x = _clamp(new_x, -ARENA_HALF, ARENA_HALF)
            new_y = _clamp(new_y, -ARENA_HALF, ARENA_HALF)

            obs.getField("translation").setSFVec3f([new_x, new_y, pos[2]])

#!/usr/bin/env python3
"""
nav_env.py — ROS2 ↔ Gymnasium Wrapper for TurtleBot3 Navigation (Webots)
=========================================================================
Stage 1 deliverable for the Autonomous Mobile Robot Navigation project.

Observation Space (74D):
    - Frame-stacked LiDAR: 24 bins × 3 frames = 72 floats  (normalized [0, 1])
    - Goal vector: [D_goal_normalized, θ_goal_normalized]    (2 floats)

Action Spaces:
    - Continuous (DDPG / TD3 / SAC): Box([-1, -1], [1, 1])
      → scaled to [linear_vel ∈ [-V_max, V_max], angular_vel ∈ [-ω_max, ω_max]]
    - Discrete  (DQN):               Discrete(8)
      → mapped to 8 predefined (linear, angular) velocity pairs (incl. reverse)

Reward:
    R_total = R_goal(+150) + R_collision(-150) + R_progress(200×Δd)
            + R_step(-2) + R_heading(1.5·cos θ) + R_angular(-0.5·|ω/ω_max|)
            + R_proximity(-3) + R_stuck(-5)

Dependencies:
    gymnasium, numpy, rclpy, sensor_msgs, nav_msgs, geometry_msgs
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------- ROS2 imports (lazy-guarded for unit-test friendliness) ----------
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist, Point
    from std_srvs.srv import Trigger
    from std_msgs.msg import Float32MultiArray

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # fallback so the class definition doesn't crash


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def euler_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation about Z) from a quaternion. Returns radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ═══════════════════════════════════════════════════════════════════════════════
# ROS2 Sensor Node (composed, NOT inherited by the Gym env)
# ═══════════════════════════════════════════════════════════════════════════════

class _SensorNode(Node):
    """
    Lightweight ROS2 node that:
      • subscribes to /scan and /odom
      • publishes to /cmd_vel
      • provides blocking wait_for_observations()
    """

    def __init__(self, node_name: str = "nav_env_node"):
        super().__init__(node_name)

        # ── QoS profile (sensor-grade: best-effort, keep-last-1) ──
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ──
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self._scan_cb, sensor_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/diffdrive_controller/odom", self._odom_cb, sensor_qos
        )

        # ── Publishers ──
        self.cmd_pub = self.create_publisher(Twist, "/diffdrive_controller/cmd_vel_unstamped", 10)
        self.goal_pub = self.create_publisher(Point, "/goal_pose", 1)

        # ── Obstacle positions subscriber ──
        self.obstacle_positions: list = []  # [(x1,y1), (x2,y2), ...]
        self.obs_pos_sub = self.create_subscription(
            Float32MultiArray, '/obstacle_positions',
            self._obs_pos_cb, 1
        )

        # ── Ground truth position subscriber ──
        # Uses supervisor's world-frame position instead of drifting odom
        self.latest_ground_truth: Optional[Point] = None
        self._gt_fresh = False
        self.gt_sub = self.create_subscription(
            Point, '/robot_ground_truth', self._gt_cb, 1
        )

        # ── Latest sensor data ──
        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None

        # ── Freshness flags (reset before each blocking wait) ──
        self._scan_fresh = False
        self._odom_fresh = False

        self.get_logger().info("🤖  NavEnv sensor node initialised.")

    # ── Callbacks ──────────────────────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self._scan_fresh = True

    def _odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self._odom_fresh = True

    def _gt_cb(self, msg: Point) -> None:
        """Ground truth world position from supervisor."""
        self.latest_ground_truth = msg
        self._gt_fresh = True

    def _obs_pos_cb(self, msg) -> None:
        """Parse flat [x1,y1,x2,y2,...] into list of (x,y) tuples."""
        data = list(msg.data)
        self.obstacle_positions = [
            (data[i], data[i + 1]) for i in range(0, len(data), 2)
        ]

    # ── Blocking wait ─────────────────────────────────────────────────────
    def wait_for_observations(self, timeout_sec: float = 5.0) -> bool:
        """
        Spin until BOTH a fresh /scan AND /odom arrive.
        Returns True on success, False on timeout.
        """
        self._scan_fresh = False
        self._odom_fresh = False
        self._gt_fresh = False
        t0 = time.monotonic()

        while not (self._scan_fresh and self._odom_fresh):
            rclpy.spin_once(self, timeout_sec=0.005)
            if (time.monotonic() - t0) > timeout_sec:
                self.get_logger().warn(
                    f"⏱  Sensor timeout after {timeout_sec:.1f}s  "
                    f"(scan={self._scan_fresh}, odom={self._odom_fresh})"
                )
                return False
        return True

    # ── Action publishing ─────────────────────────────────────────────────
    # TurtleBot3 Burger motor limits
    _WHEEL_RADIUS = 0.033      # metres
    _WHEEL_SEP    = 0.160      # metres (track width)
    _MAX_WHEEL_VEL = 6.0       # rad/s (motor limit 6.67, use 6.0 for margin)

    def publish_velocity(self, linear: float, angular: float) -> None:
        """Publish velocity, clamping to stay within motor limits.

        The diff-drive kinematics convert (v, ω) to wheel velocities:
            ω_left  = (v - ω·L/2) / r
            ω_right = (v + ω·L/2) / r
        We scale down (v, ω) proportionally if either wheel exceeds max.
        """
        r = self._WHEEL_RADIUS
        half_L = self._WHEEL_SEP / 2.0

        # Compute required wheel velocities
        wl = (linear - angular * half_L) / r
        wr = (linear + angular * half_L) / r

        # Scale down if either exceeds motor limit
        max_w = max(abs(wl), abs(wr))
        if max_w > self._MAX_WHEEL_VEL:
            scale = self._MAX_WHEEL_VEL / max_w
            linear *= scale
            angular *= scale

        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop_robot(self) -> None:
        self.publish_velocity(0.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Gymnasium Environment
# ═══════════════════════════════════════════════════════════════════════════════

class NavigationEnv(gym.Env):
    """
    Custom Gymnasium environment bridging ROS2 / Webots TurtleBot3 navigation
    with Deep RL algorithms (DQN, DDPG, TD3, SAC).

    Parameters
    ----------
    discrete_action : bool
        If True → Discrete(5) action space for DQN.
        If False → Box(2,) continuous action space for DDPG / TD3 / SAC.
    goal_position : tuple[float, float] | None
        Fixed (x, y) goal. If None, a random goal is sampled on each reset.
    max_episode_steps : int
        Hard step limit per episode.
    collision_threshold : float
        Minimum LiDAR reading (metres) that triggers a collision.
    goal_threshold : float
        Euclidean distance (metres) to consider the goal reached.
    v_max : float
        Maximum linear velocity (m/s).
    w_max : float
        Maximum angular velocity (rad/s).
    max_lidar_range : float
        Sensor max range (metres) — used for normalisation.
    max_goal_distance : float
        Upper bound on expected goal distance — used for normalisation.
    sensor_timeout : float
        Seconds to wait for fresh sensor data before raising.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    # ── Discrete action lookup table ──
    # Index → (linear_vel_fraction, angular_vel_fraction)
    # Fractions ∈ [-1,1]; scaled by v_max / w_max.
    DISCRETE_ACTIONS = {
        0: (0.05 / 0.22, 1.0),    # Sharp left    – slow + max-left
        1: (0.15 / 0.22, 0.5),    # Forward-left  – medium + gentle-left
        2: (1.0, 0.0),            # Straight      – max forward
        3: (0.15 / 0.22, -0.5),   # Forward-right – medium + gentle-right
        4: (0.05 / 0.22, -1.0),   # Sharp right   – slow + max-right
        5: (-0.5, 0.0),           # Reverse       – back up slowly
        6: (-0.3, 0.8),           # Reverse-left  – back + turn left
        7: (-0.3, -0.8),          # Reverse-right – back + turn right
    }

    def __init__(
        self,
        discrete_action: bool = False,
        goal_position: Optional[Tuple[float, float]] = None,
        max_episode_steps: int = 500,
        collision_threshold: float = 0.28,
        goal_threshold: float = 0.35,
        v_max: float = 0.22,
        w_max: float = 2.0,
        max_lidar_range: float = 3.5,
        max_goal_distance: float = 10.0,
        sensor_timeout: float = 5.0,
        num_lidar_bins: int = 24,
        num_frames: int = 3,
    ):
        super().__init__()

        # ── Config ────────────────────────────────────────────────────────
        self.discrete_action = discrete_action
        self.goal_position = goal_position
        self.max_episode_steps = max_episode_steps
        self.collision_threshold = collision_threshold
        self.goal_threshold = goal_threshold
        self.v_max = v_max
        self.w_max = w_max
        self.max_lidar_range = max_lidar_range
        self.max_goal_distance = max_goal_distance
        self.sensor_timeout = sensor_timeout
        self.num_lidar_bins = num_lidar_bins
        self.num_frames = num_frames

        # ── Observation space: 24 bins × 3 frames + 2 goal = 74 ──────────
        self.obs_dim = self.num_lidar_bins * self.num_frames + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # ── Action space ──────────────────────────────────────────────────
        if self.discrete_action:
            self.action_space = spaces.Discrete(8)  # 5 forward + 3 reverse
        else:
            # Normalised continuous actions ∈ [-1,1] × [-1,1]
            # action[0] ∈ [-1,1]: negative=reverse, positive=forward
            # action[1] ∈ [-1,1]: angular velocity
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
            )

        # ── Frame-stack buffer (24-bin LiDAR per frame) ───────────────────
        self._lidar_stack: deque[np.ndarray] = deque(maxlen=self.num_frames)

        # ── Episode state ─────────────────────────────────────────────────
        self._step_count: int = 0
        self._prev_goal_dist: float = 0.0
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_yaw: float = 0.0
        self._goal_x: float = 0.0
        self._goal_y: float = 0.0
        self._done: bool = False

        # ── Stuck detection (for square arena corner recovery) ────────────
        self._position_history: deque = deque(maxlen=50)
        self._stuck_threshold: float = 0.10  # metres moved in 50 steps

        # ── Phase-aware reward coefficients ────────────────────────────────
        # Read curriculum phase from env var (set before launching Webots).
        # Adjusts reward weights so the agent learns goal-seeking in Phase 1
        # and obstacle avoidance in Phases 2-4.
        self._phase = int(os.environ.get("CURRICULUM_PHASE", "1"))
        self._reward_cfg = {
            # Phase 1 — Empty Room: strong heading, penalise spinning
            1: {"heading": 1.0, "angular": -0.2, "proximity": -3.0,
                "stuck": -5.0, "progress": 50.0},
            # Phase 2 — Static Obstacles: softer heading, allow some turns
            2: {"heading": 0.5, "angular": -0.1, "proximity": -0.5,
                "stuck": -0.5, "progress": 50.0},
            # Phase 3 — Slow Dynamic: weak heading, soft turn penalty
            3: {"heading": 0.3, "angular": -0.05, "proximity": -1.0,
                "stuck": -0.5, "progress": 50.0},
            # Phase 4 — Fast Dynamic: weak heading, soft turn penalty
            4: {"heading": 0.3, "angular": -0.05, "proximity": -1.5,
                "stuck": -0.5, "progress": 50.0},
        }.get(self._phase, {"heading": 1.0, "angular": -0.2,
                            "proximity": -3.0, "stuck": -5.0,
                            "progress": 50.0})

        # ── ROS2 node ────────────────────────────────────────────────────
        self._node: Optional[_SensorNode] = None
        self._ros_initialised: bool = False
        self._init_ros2()

    # ══════════════════════════════════════════════════════════════════════
    # ROS2 lifecycle
    # ══════════════════════════════════════════════════════════════════════

    def _init_ros2(self) -> None:
        """Start rclpy (if needed) and create the sensor node."""
        if not ROS2_AVAILABLE:
            print("[NavigationEnv] ⚠  rclpy not found — running in HEADLESS mode. "
                  "Sensor data will be synthetic zeros.")
            return

        if not rclpy.ok():
            rclpy.init()

        self._node = _SensorNode()
        self._ros_initialised = True

    # ══════════════════════════════════════════════════════════════════════
    # Sensor processing
    # ══════════════════════════════════════════════════════════════════════

    def _process_scan(self, scan_msg) -> np.ndarray:
        """
        Downsample a raw LaserScan into ``num_lidar_bins`` normalised floats.

        Strategy:
            1. Replace inf / NaN with max_lidar_range.
            2. Partition the 360° sweep into ``num_lidar_bins`` equal sectors.
            3. Each bin = **min** reading in its sector (conservatively closest obstacle).
            4. Normalise by max_lidar_range → [0, 1].
        """
        ranges = np.array(scan_msg.ranges, dtype=np.float32)

        # Sanitise: replace inf/NaN with max range, clamp to [0, max]
        ranges = np.where(np.isfinite(ranges), ranges, self.max_lidar_range)
        ranges = np.clip(ranges, 0.0, self.max_lidar_range)

        num_raw = len(ranges)
        sector_size = max(1, num_raw // self.num_lidar_bins)  # guard: at least 1
        bins = np.empty(self.num_lidar_bins, dtype=np.float32)

        for i in range(self.num_lidar_bins):
            start = i * sector_size
            end = min(start + sector_size, num_raw)  # guard: don't exceed array
            if start < num_raw:
                bins[i] = np.min(ranges[start:end])
            else:
                bins[i] = 1.0  # max range (safe default)

        # Normalise to [0, 1]
        bins /= self.max_lidar_range
        # Final sanitise — ensure no NaN/inf survives
        bins = np.where(np.isfinite(bins), bins, 1.0)
        return bins

    def _update_pose_from_odom(self, odom_msg) -> None:
        """Extract robot pose: x, y, yaw from ground truth (preferred) or odom (fallback).

        Uses supervisor ground truth for (x, y, yaw) because odometry
        accumulates across teleports and drifts from the real world
        position.  Odom is still subscribed to for sync/timing and as fallback.
        """
        # ── Prefer ground truth (world frame) ──
        if (self._ros_initialised and self._node is not None
                and self._node.latest_ground_truth is not None):
            gt = self._node.latest_ground_truth
            # Guard: reject NaN ground truth values
            if math.isfinite(gt.x) and math.isfinite(gt.y):
                self._robot_x = gt.x
                self._robot_y = gt.y
                self._robot_yaw = gt.z if math.isfinite(gt.z) else 0.0
            else:
                # Ground truth is NaN — keep previous valid values
                pass
        else:
            # Fallback to odom if ground truth not available yet
            pos = odom_msg.pose.pose.position
            ori = odom_msg.pose.pose.orientation
            x, y = pos.x, pos.y
            if math.isfinite(x) and math.isfinite(y):
                self._robot_x = x
                self._robot_y = y
                yaw = euler_from_quaternion(ori.x, ori.y, ori.z, ori.w)
                self._robot_yaw = yaw if math.isfinite(yaw) else 0.0

    def _compute_goal_vector(self) -> Tuple[float, float]:
        """
        Returns (D_goal_normalised, θ_goal_normalised)
            D_goal: Euclidean distance / max_goal_distance  → [0, 1]
            θ_goal: signed angle to goal / π                → [-1, 1]
        """
        dx = self._goal_x - self._robot_x
        dy = self._goal_y - self._robot_y

        # NaN guard: if robot position is NaN, return safe defaults
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return 0.5, 0.5

        distance = math.hypot(dx, dy)

        # Angle from robot to goal in world frame
        angle_to_goal = math.atan2(dy, dx)
        # Relative to robot heading
        yaw = self._robot_yaw if math.isfinite(self._robot_yaw) else 0.0
        relative_angle = normalize_angle(angle_to_goal - yaw)

        d_norm = np.clip(distance / self.max_goal_distance, 0.0, 1.0)
        theta_norm = np.clip(relative_angle / math.pi, -1.0, 1.0)
        # Shift θ from [-1,1] → [0,1] to keep entire obs box in [0,1]
        theta_norm_shifted = (theta_norm + 1.0) / 2.0

        return float(d_norm), float(theta_norm_shifted)

    # ══════════════════════════════════════════════════════════════════════
    # Observation assembly
    # ══════════════════════════════════════════════════════════════════════

    def _build_observation(self) -> np.ndarray:
        """
        Concatenate:
            [frame_{t-2} (24) | frame_{t-1} (24) | frame_t (24) | D_goal | θ_goal]
        Total = 74 floats, all in [0, 1].
        """
        # Flatten frame stack
        stacked = np.concatenate(list(self._lidar_stack))  # (72,)

        # Goal vector
        d_goal, theta_goal = self._compute_goal_vector()
        goal_vec = np.array([d_goal, theta_goal], dtype=np.float32)

        obs = np.concatenate([stacked, goal_vec]).astype(np.float32)

        # Sanitise entire observation — never let NaN reach the model
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        assert obs.shape == (self.obs_dim,), f"Obs shape mismatch: {obs.shape}"
        return obs

    # ══════════════════════════════════════════════════════════════════════
    # Reward computation
    # ══════════════════════════════════════════════════════════════════════

    def _current_goal_distance(self) -> float:
        """Euclidean distance to goal, NaN-safe."""
        dx = self._goal_x - self._robot_x
        dy = self._goal_y - self._robot_y
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return self._prev_goal_dist if math.isfinite(self._prev_goal_dist) else 1.0
        return math.hypot(dx, dy)

    def _check_collision(self) -> bool:
        """True if any LiDAR bin in the CURRENT frame ≤ collision_threshold."""
        if len(self._lidar_stack) == 0:
            return False
        current_frame = self._lidar_stack[-1]  # normalised [0,1]
        # Convert back to metres for threshold comparison
        min_reading = np.min(current_frame) * self.max_lidar_range
        return min_reading <= self.collision_threshold

    def _check_goal_reached(self) -> bool:
        return self._current_goal_distance() <= self.goal_threshold

    def _compute_reward(self) -> Tuple[float, bool, bool]:
        """
        Phase-aware reward:
            R_total = R_goal + R_collision + R_progress + R_step
                    + R_heading + R_angular + R_proximity + R_stuck

        Coefficients scale with curriculum phase:
          Phase 1 (empty):   strong heading, penalise spinning
          Phase 2 (static):  softer heading, allow evasive turns
          Phase 3 (slow):    weak heading, no turn penalty
          Phase 4 (fast):    weak heading, strong obstacle avoidance

        Returns: (reward, terminated, truncated)
        """
        cfg = self._reward_cfg
        terminated = False
        truncated = False
        reward = 0.0

        # ── R_collision (-300, terminal) ──
        if self._check_collision():
            reward += -300.0
            terminated = True
            return reward, terminated, truncated

        # ── R_goal (+250, terminal) ──
        if self._check_goal_reached():
            reward += 250.0
            terminated = True
            return reward, terminated, truncated

        # ── R_progress (phase-scaled) ──
        current_dist = self._current_goal_distance()
        if not math.isfinite(current_dist):
            current_dist = self._prev_goal_dist if math.isfinite(self._prev_goal_dist) else 1.0
        if not math.isfinite(self._prev_goal_dist):
            self._prev_goal_dist = current_dist  # repair broken chain
        delta = self._prev_goal_dist - current_dist
        if not math.isfinite(delta):
            delta = 0.0
        progress = delta * cfg["progress"]
        # Clamp progress reward to prevent explosion
        progress = max(-20.0, min(20.0, progress))
        reward += progress
        self._prev_goal_dist = current_dist

        # ── R_step (time penalty) ──
        reward += -1.5

        # ── R_heading (face-the-goal bonus / anti-spiral) ──
        dx = self._goal_x - self._robot_x
        dy = self._goal_y - self._robot_y
        if math.isfinite(dx) and math.isfinite(dy) and (abs(dx) + abs(dy)) > 1e-6:
            angle_to_goal = math.atan2(dy, dx)
            yaw = self._robot_yaw if math.isfinite(self._robot_yaw) else 0.0
            relative_angle = normalize_angle(angle_to_goal - yaw)
            heading_reward = cfg["heading"] * math.cos(relative_angle)
            if math.isfinite(heading_reward):
                reward += heading_reward
        # else: skip heading reward entirely when position is invalid

        # ── R_angular (excessive-rotation penalty / anti-spiral) ──
        if hasattr(self, '_last_angular_vel') and cfg["angular"] != 0.0:
            ang_ratio = abs(self._last_angular_vel) / self.w_max
            penalty = cfg["angular"] * ang_ratio
            if math.isfinite(penalty):
                reward += penalty

        # ── R_proximity (wall/obstacle danger zone penalty) ──
        if len(self._lidar_stack) > 0:
            min_dist = float(np.min(self._lidar_stack[-1])) * self.max_lidar_range
            if math.isfinite(min_dist) and min_dist < 0.35 and min_dist > self.collision_threshold:
                reward += (cfg["proximity"] * 0.1)

        # ── R_stuck (corner/obstacle recovery penalty) ──
        self._position_history.append((self._robot_x, self._robot_y))
        if len(self._position_history) >= 50:
            old_x, old_y = self._position_history[0]
            if math.isfinite(old_x) and math.isfinite(old_y):
                dist_moved = math.hypot(
                    self._robot_x - old_x, self._robot_y - old_y
                )
                if math.isfinite(dist_moved) and dist_moved < self._stuck_threshold:
                    reward += (cfg["stuck"] * 0.1)

        # ── Final NaN guard on total reward ──
        if not math.isfinite(reward):
            reward = -50.0

        # ── Truncation check ──
        if self._step_count >= self.max_episode_steps:
            truncated = True

        return reward, terminated, truncated

    # ══════════════════════════════════════════════════════════════════════
    # Action mapping
    # ══════════════════════════════════════════════════════════════════════

    def _map_action(self, action) -> Tuple[float, float]:
        """
        Convert any incoming action to (linear_vel, angular_vel) in SI.

        Discrete(int) → look up table → (v, w)
        Box([0,1], [-1,1]) → scale to (v_max, w_max)
        """
        if self.discrete_action:
            # Safety: clamp to valid index
            idx = int(action)
            idx = max(0, min(idx, len(self.DISCRETE_ACTIONS) - 1))
            v_frac, w_frac = self.DISCRETE_ACTIONS[idx]
            return v_frac * self.v_max, w_frac * self.w_max
        else:
            # Continuous: action[0] ∈ [-1,1] → linear (negative=reverse)
            #             action[1] ∈ [-1,1] → angular
            action = np.asarray(action, dtype=np.float32).flatten()
            # Guard: if model outputs NaN (corrupted weights), stop the robot
            if not np.all(np.isfinite(action)):
                return 0.0, 0.0
            linear = float(np.clip(action[0], -1.0, 1.0)) * self.v_max
            angular = float(np.clip(action[1], -1.0, 1.0)) * self.w_max
            return linear, angular

    # ══════════════════════════════════════════════════════════════════════
    # Goal management
    # ══════════════════════════════════════════════════════════════════════

    def _sample_goal(self) -> None:
        """Set goal position — fixed if given, else random within arena.
        Enforces:
          - minimum 1.0m distance from robot (no free successes)
          - minimum 0.5m distance from all obstacles (goal not inside obstacle)
        """
        if self.goal_position is not None:
            self._goal_x, self._goal_y = self.goal_position
        else:
            min_goal_dist = 1.0   # from robot
            min_obs_dist = 0.5    # from any obstacle
            # Get obstacle positions from subscriber
            obs_list = []
            if self._ros_initialised and hasattr(self._node, 'obstacle_positions'):
                obs_list = self._node.obstacle_positions

            for _ in range(100):
                gx = float(np.random.uniform(-1.0, 1.0))
                gy = float(np.random.uniform(-1.0, 1.0))
                # Check distance to robot
                if math.hypot(gx - self._robot_x, gy - self._robot_y) < min_goal_dist:
                    continue
                # Check distance to all obstacles
                too_close = False
                for ox, oy in obs_list:
                    if math.hypot(gx - ox, gy - oy) < min_obs_dist:
                        too_close = True
                        break
                if not too_close:
                    self._goal_x, self._goal_y = gx, gy
                    break
            else:
                # Fallback: place goal on opposite side of arena
                self._goal_x = -self._robot_x if abs(self._robot_x) > 0.1 else 1.0
                self._goal_y = -self._robot_y if abs(self._robot_y) > 0.1 else 1.0

        # Publish goal position so the supervisor can move the green marker
        if self._ros_initialised and hasattr(self._node, 'goal_pub'):
            msg = Point()
            msg.x = self._goal_x
            msg.y = self._goal_y
            msg.z = 0.0
            self._node.goal_pub.publish(msg)

    # ══════════════════════════════════════════════════════════════════════
    # Simulation reset helpers
    # ══════════════════════════════════════════════════════════════════════

    def _reset_simulation(self) -> None:
        """
        Reset the Webots simulation via the /reset_robot service
        provided by supervisor_controller.py.

        The supervisor teleports the robot to start and randomises obstacles.
        """
        if not self._ros_initialised:
            return

        # Stop robot motion immediately
        self._node.stop_robot()

        # Call the supervisor's /reset_robot service
        if not hasattr(self, '_reset_client'):
            self._reset_client = self._node.create_client(
                Trigger, '/reset_robot'
            )

        # Retry up to 3 times with generous timeout for Fast Mode
        reset_ok = False
        for attempt in range(3):
            if self._reset_client.wait_for_service(timeout_sec=10.0):
                future = self._reset_client.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(
                    self._node, future, timeout_sec=10.0
                )
                if future.result() is not None:
                    reset_ok = True
                    break
                else:
                    self._node.get_logger().warn(
                        f"⚠  Reset call failed (attempt {attempt+1}/3)"
                    )
            else:
                self._node.get_logger().warn(
                    f"⚠  /reset_robot service not available "
                    f"(attempt {attempt+1}/3)"
                )

        if not reset_ok:
            self._node.get_logger().error(
                "❌  All reset attempts failed — supervisor may be down"
            )

        # Give the simulation time to settle after reset
        time.sleep(0.3)

    def _wait_initial_data(self) -> bool:
        """Spin until the first /scan and /odom arrive after reset."""
        if not self._ros_initialised:
            return True

        return self._node.wait_for_observations(timeout_sec=self.sensor_timeout)

    # ══════════════════════════════════════════════════════════════════════
    # Gymnasium API
    # ══════════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        1. Reset simulation & teleport robot.
        2. Wait for first sensor data (so _robot_x/_robot_y update).
        3. Sample a new goal (needs current robot pos for min distance).
        4. Initialise 3-frame LiDAR stack (replicate first scan ×3).
        5. Return initial observation.
        """
        super().reset(seed=seed)

        # ── Reset simulation ──────────────────────────────────────────────
        self._reset_simulation()

        # ── Episode counters ──────────────────────────────────────────────
        self._step_count = 0
        self._done = False
        self._position_history.clear()

        # ── Wait for fresh sensor data (updates _robot_x, _robot_y) ──────
        if self._ros_initialised:
            success = self._wait_initial_data()
            if not success:
                print("[NavigationEnv] ⚠  Timeout on initial sensor data — "
                      "using zeros.")

        # ── Goal (AFTER odom so min-distance check works) ─────────────────
        self._sample_goal()

        # ── Process first scan & warm the frame stack ────────────────────
        if self._ros_initialised and self._node.latest_scan is not None:
            first_frame = self._process_scan(self._node.latest_scan)
        else:
            # Headless / timeout fallback: empty frame
            first_frame = np.ones(self.num_lidar_bins, dtype=np.float32)

        # Replicate to fill the 3-frame history
        self._lidar_stack.clear()
        for _ in range(self.num_frames):
            self._lidar_stack.append(first_frame.copy())

        # ── Pose from odom ───────────────────────────────────────────────
        if self._ros_initialised and self._node.latest_odom is not None:
            self._update_pose_from_odom(self._node.latest_odom)

        # ── Initial goal distance (NaN-safe) ────────────────────────────
        dist = self._current_goal_distance()
        self._prev_goal_dist = dist if math.isfinite(dist) else 1.0

        obs = self._build_observation()
        # Sanitise initial observation
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        info: Dict[str, Any] = {
            "goal": (self._goal_x, self._goal_y),
            "robot_pose": (self._robot_x, self._robot_y, self._robot_yaw),
            "goal_distance": self._prev_goal_dist,
        }

        return obs, info

    def step(
        self, action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step:

        1. Map action → (linear_vel, angular_vel).
        2. Publish to /cmd_vel.
        3. Block until fresh /scan + /odom arrive.
        4. Update LiDAR frame stack (shift & append).
        5. Compute reward, check termination.
        6. Return (obs, reward, terminated, truncated, info).
        """
        assert not self._done, "Episode already terminated — call reset()."

        self._step_count += 1

        # ── 1. Map action ────────────────────────────────────────────────
        linear_vel, angular_vel = self._map_action(action)
        self._last_angular_vel = angular_vel  # for R_angular penalty

        # ── 2. Publish velocity command ──────────────────────────────────
        if self._ros_initialised:
            self._node.publish_velocity(linear_vel, angular_vel)

        # ── 3. Block until new sensor data arrives ───────────────────────
        if self._ros_initialised:
            self._node.wait_for_observations(timeout_sec=self.sensor_timeout)

        # ── 4. Process new scan → update frame stack ─────────────────────
        if self._ros_initialised and self._node.latest_scan is not None:
            new_frame = self._process_scan(self._node.latest_scan)
        else:
            new_frame = np.ones(self.num_lidar_bins, dtype=np.float32)

        self._lidar_stack.append(new_frame)  # deque auto-pops oldest

        # ── Update pose from odom ────────────────────────────────────────
        if self._ros_initialised and self._node.latest_odom is not None:
            self._update_pose_from_odom(self._node.latest_odom)

        # ── PHYSICS CRASH DETECTION ──────────────────────────────────────
        # If robot position is NaN or far outside the arena, the physics
        # engine has broken. Force-terminate with safe values to prevent
        # NaN from corrupting the neural network weights.
        physics_crashed = False
        if (not np.isfinite(self._robot_x) or not np.isfinite(self._robot_y)
                or abs(self._robot_x) > 5.0 or abs(self._robot_y) > 5.0
                or not np.isfinite(self._robot_yaw)):
            physics_crashed = True
            print(f"[NavEnv] ⚠ PHYSICS CRASH DETECTED at step {self._step_count}: "
                  f"pos=({self._robot_x:.2f}, {self._robot_y:.2f})")
            # Reset to safe values
            self._robot_x = 0.0
            self._robot_y = 0.0
            self._robot_yaw = 0.0

        if physics_crashed:
            # Return safe, finite values — episode ends immediately
            self._done = True
            if self._ros_initialised:
                self._node.stop_robot()
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = -50.0  # moderate penalty, not catastrophic
            info = {
                "goal": (self._goal_x, self._goal_y),
                "robot_pose": (0.0, 0.0, 0.0),
                "goal_distance": 1.0,
                "collision": True,
                "goal_reached": False,
                "step": self._step_count,
                "linear_vel": 0.0,
                "angular_vel": 0.0,
                "physics_crash": True,
            }
            return obs, reward, True, False, info

        # ── 5. Reward & termination ──────────────────────────────────────
        reward, terminated, truncated = self._compute_reward()

        # Sanitise reward — never let NaN reach the neural network
        if not np.isfinite(reward):
            print(f"[NavEnv] ⚠ NaN reward detected at step {self._step_count}, using -50.0")
            reward = -50.0

        if terminated or truncated:
            self._done = True
            if self._ros_initialised:
                self._node.stop_robot()

        # ── 6. Build observation & info ──────────────────────────────────
        obs = self._build_observation()

        # Sanitise observation — replace any NaN/inf with 0
        if not np.all(np.isfinite(obs)):
            print(f"[NavEnv] ⚠ NaN in observation at step {self._step_count}, sanitising")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        info: Dict[str, Any] = {
            "goal": (self._goal_x, self._goal_y),
            "robot_pose": (self._robot_x, self._robot_y, self._robot_yaw),
            "goal_distance": self._current_goal_distance(),
            "collision": self._check_collision(),
            "goal_reached": self._check_goal_reached(),
            "step": self._step_count,
            "linear_vel": linear_vel,
            "angular_vel": angular_vel,
        }

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up ROS2 resources."""
        if self._ros_initialised and self._node is not None:
            self._node.stop_robot()
            self._node.destroy_node()
            self._node = None

        if ROS2_AVAILABLE and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

        self._ros_initialised = False

    # ══════════════════════════════════════════════════════════════════════
    # Rendering (optional, for debugging)
    # ══════════════════════════════════════════════════════════════════════

    def render(self) -> None:
        """Print a compact status line (Webots handles visual rendering)."""
        if len(self._lidar_stack) > 0:
            min_lidar = np.min(self._lidar_stack[-1]) * self.max_lidar_range
        else:
            min_lidar = -1.0

        print(
            f"[Step {self._step_count:>4d}]  "
            f"Pose=({self._robot_x:+.2f}, {self._robot_y:+.2f}, "
            f"yaw={math.degrees(self._robot_yaw):+.1f}°)  "
            f"Goal_D={self._current_goal_distance():.2f}m  "
            f"Min_LiDAR={min_lidar:.2f}m"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Quick self-test (run without ROS2 to verify shapes & logic)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  NavigationEnv — Stage 1 Self-Test (headless, no ROS2)")
    print("=" * 72)

    for mode_name, discrete in [("CONTINUOUS", False), ("DISCRETE (8-action)", True)]:
        print(f"\n{'─'*36}")
        print(f"  Mode: {mode_name}")
        print(f"{'─'*36}")

        env = NavigationEnv(
            discrete_action=discrete,
            goal_position=(2.0, 2.0),
            max_episode_steps=10,
        )

        print(f"  Observation space : {env.observation_space}")
        print(f"  Action space      : {env.action_space}")
        print(f"  Obs dim           : {env.obs_dim}")

        obs, info = env.reset()
        print(f"  Reset obs shape   : {obs.shape}")
        print(f"  Reset info        : {info}")
        assert obs.shape == (74,), f"Expected (74,), got {obs.shape}"

        # Run a few random steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: action={action}  reward={reward:+.2f}  "
                  f"terminated={terminated}  truncated={truncated}")
            assert obs.shape == (74,)
            if terminated or truncated:
                obs, info = env.reset()

        env.close()
        print(f"  ✅  {mode_name} mode passed.")

    print("\n" + "=" * 72)
    print("  All self-tests passed! 🎉")
    print("=" * 72)

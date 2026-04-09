# 3-Day Autonomous Robot Navigation Implementation Plan

This plan breaks down the DRL Navigation project into manageable subproblems spread over a 3-day execution timeline, aiming to build, train, and test the DDPG, DQN, and TD3 algorithms in a Webots ROS2 environment.

## User Review Required

> [!IMPORTANT]
> Please review the 3-day schedule and subproblem breakdown. Let me know if the prioritization aligns exactly with your timeframe and if any particular section (such as the focus on dynamic obstacles) needs more time allocated.

## Proposed Changes / 3-Day Execution Timeline

### Day 1: Environment Foundation & ROS2 Bridge
**Goal:** Establish a stable communication pipeline between Webots, ROS2 Humble, and a custom Gymnasium wrapper.
**Subproblems:**
1. **Workspace Initialization:** Set up the ROS2 workspace locally with `webots_ros2`, `stable-baselines3`, `torch`, and `gymnasium` inside WSL2.
2. **Gym Wrapper Structure (`nav_env.py`):** Create the custom base `gymnasium.Env` logic and embed a ROS2 `rclpy` node to handle asynchronous callbacks.
3. **Pub/Sub Integration:** Connect the `/cmd_vel` topic for agent actions and subscribe to `/scan` (LiDAR) and `/odom` (Odometry) for states.
4. **Simulation Control:** Implement a robust `reset()` behavior (teleporting the TurtleBot3 to start coordinates and updating goal waypoints) and ensure proper synchronization during `step()`.

### Day 2: State/Action Spaces & Reward Engineering
**Goal:** Formalize the environment's observations, action boundaries, and learning incentives.
**Subproblems:**
1. **LiDAR State Processing:** Process raw 360-degree `/scan` data into 24 normalized bins. Calculate the relative goal distance ($D_{goal}$) and angle ($\theta_{goal}$) to complete the 26-dimensional state space.
2. **Action Space Definition:** 
   - Formulate the continuous 2D space `[linear_vel, angular_vel]` for DDPG and TD3.
   - Formulate the discrete 5-action space (sharp left, forward-left, straight, forward-right, sharp right) for DQN.
3. **Reward Function Construction:**
   - Goal completion ($+100$, terminal state).
   - Collision penalty ($-100$, terminal state).
   - Dense progress reward (positive scalar for reducing distance to the goal).
   - Step penalty ($-1$ per step to encourage faster navigation).
4. **Sanity Check:** Run a random-agent policy to ensure the simulation steps cleanly, callbacks update reliably, and state/reward values output correctly without deadlocking the ROS2 executor.

### Day 3: Algorithm Training, Tuning, & Dynamic Enhancements
**Goal:** Train the baseline DRL models, analyze their performance, and explore dynamic obstacles.
**Subproblems:**
1. **Algorithm Initialization:** Set up the Multilayer Perceptrons (MLPs) for DQN, DDPG, and TD3 using `stable-baselines3`.
2. **Baseline Training:** 
   - Train the TD3 model (~50,000 steps for fast convergence).
   - Train DQN and DDPG models (~100,000 steps with identical seeds for accurate comparison).
3. **Evaluation Logging:** Record both "Episode Reward" and "Episode Length" metrics directly to TensorBoard.
4. **Dynamic Obstacles Integration (Research Application):** Begin scaffolding frame-stacking (using LSTMs or observation history buffers) and evaluate performance degradation when target/obstacle velocity changes.

## Open Questions

> [!NOTE]
> 1. Should we deploy **Curriculum Learning** right away on Day 3 for the dynamic obstacles, or would you prefer dedicating Day 3 strictly to static baseline comparisons first?
> 2. Are all required dependencies (Webots, ROS2 Humble, Stable Baselines 3) already installed in your WSL2 environment, or should I allocate dedicated time on Day 1 for dependency troubleshooting?
> 3. Does this adequately cover what you need to prompt Claude/Opus, or do you want me to format this directly as a markdown prompt string that you can copy/paste to another model?

## Verification Plan

### Automated Tests
- Successful execution of random-action episodes on Day 2 without simulation crashes or node timeouts.
- Clean TensorBoard metrics visualization (Reward/Episode Length curves) logging efficiently on Day 3.

### Manual Verification
- Verifying the TurtleBot3 safely navigates around basic obstacles visually within the Webots viewer.

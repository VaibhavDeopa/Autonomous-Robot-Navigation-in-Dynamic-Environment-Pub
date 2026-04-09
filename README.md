<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Stable--Baselines3-2.x-00C853?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Webots-R2025a-007ACC?logo=webots&logoColor=white" />
  <img src="https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros&logoColor=white" />
  <img src="https://img.shields.io/badge/Gymnasium-0.29+-4B8BBE?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia&logoColor=white" />
</p>

#  Autonomous Robot Navigation in Dynamic Environment

> **A Deep Reinforcement Learning framework for training a TurtleBot3 Burger to navigate from a random start to a random goal in a 3×3 m arena, progressively learning to avoid static boxes and fast-moving cylindrical obstacles through a 4-phase curriculum.**

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Architecture Overview](#-architecture-overview)
- [Model Design and Implementation](#-1-model-design-and-implementation)
  - [DQN — Deep Q-Network](#dqn--deep-q-network)
  - [DDPG — Deep Deterministic Policy Gradient](#ddpg--deep-deterministic-policy-gradient)
  - [TD3 — Twin Delayed DDPG](#td3--twin-delayed-ddpg)
  - [SAC — Soft Actor-Critic (Modified)](#sac--soft-actor-critic-modified)
- [Algorithmic Behavior and Phase Policies](#-2-algorithmic-behavior-and-phase-policies)
  - [Curriculum Learning: 4-Phase Training Pipeline](#curriculum-learning-4-phase-training-pipeline)
  - [Per-Algorithm Behavioral Analysis](#per-algorithm-behavioral-analysis)
- [Reward Structure](#-3-reward-structure)
  - [Terminal Rewards](#terminal-rewards)
  - [Shaping Rewards](#shaping-rewards-per-step)
  - [Phase-Specific Reward Coefficients](#phase-specific-reward-coefficients)
- [Performance Metrics](#-4-performance-metrics)
  - [Final Model Performance Comparison](#-final-model-performance-comparison--phase-4-200-episodes)
  - [Training Convergence](#training-convergence--per-phase-breakdown)
  - [Key Observations & Analysis](#-key-observations--analysis)
- [Environment Specification](#-environment-specification)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [License](#-license)

---

##  Project Overview

This project implements a **comparative study of four Deep Reinforcement Learning algorithms** for autonomous mobile robot navigation in environments of increasing complexity. A **TurtleBot3 Burger** robot is trained inside a **Webots R2025a** physics simulator, bridged to the learning pipeline through **ROS2 Humble** and wrapped as a **Gymnasium** custom environment.

The core research question: *Which DRL algorithm best generalises goal-directed navigation skills from an empty room to an arena with fast-moving obstacles — and how does curriculum learning facilitate this transfer?*

### Key Contributions

- ✅ **4 DRL algorithms** compared under identical conditions: DQN, DDPG, TD3, SAC
- ✅ **4-phase curriculum** with progressive environment complexity  
- ✅ **Phase-aware reward shaping** — coefficients auto-adjust per curriculum stage  
- ✅ **Frame-stacked LiDAR observations** (3×24 bins) enabling temporal reasoning about moving obstacles  
- ✅ **Trajectory-based dynamic obstacles** (linear + parabolic paths) for realistic motion  
- ✅ **Physics-crash hardening** — NaN detection, automatic recovery, weight corruption prevention  

---

##  Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     WEBOTS R2025a SIMULATOR                      │
│  ┌────────────────┐  ┌──────────┐  ┌──────────────────────────┐ │
│  │ TurtleBot3     │  │ 3×3m     │  │ Supervisor Plugin        │ │
│  │ Burger         │  │ Square   │  │ • Phase-aware obstacle   │ │
│  │ • LiDAR 360°   │  │ Arena    │  │   placement & movement   │ │
│  │ • Diff-Drive   │  │ • Walls  │  │ • Robot reset/teleport   │ │
│  │ • Odometry     │  │ • Floor  │  │ • Ground truth publisher │ │
│  └───────┬────────┘  └──────────┘  └────────────┬─────────────┘ │
│          │ /scan, /odom                          │ /reset_robot  │
└──────────┼───────────────────────────────────────┼───────────────┘
           │            ROS2 Humble                │
           ▼                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    NAVIGATION ENV (Gymnasium)                     │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────┐ │
│  │ Sensor Node  │  │ Observation   │  │ Reward Engine          │ │
│  │ • /scan sub  │  │ Builder       │  │ • Phase-aware coeffs   │ │
│  │ • /odom sub  │  │ • 24-bin      │  │ • 8-component reward   │ │
│  │ • /cmd_vel   │  │   LiDAR ×3    │  │ • NaN-safe computation │ │
│  │   publisher  │  │ • Goal vector │  │                        │ │
│  └──────────────┘  └───────────────┘  └───────────────────────┘ │
│                     Output: obs ∈ ℝ⁷⁴, reward ∈ ℝ               │
└──────────────────────────────────────────────────────────────────┘
           │                                       ▲
           ▼                                       │
┌──────────────────────────────────────────────────────────────────┐
│              STABLE-BASELINES3 TRAINING LOOP                     │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────────────────────┐ │
│  │  DQN   │  │  DDPG  │  │  TD3   │  │  SAC (Modified)        │ │
│  │Discrete│  │Cont.   │  │Cont.   │  │Cont. + Auto-Entropy    │ │
│  └────────┘  └────────┘  └────────┘  └────────────────────────┘ │
│        Callbacks: Metrics · Checkpoints · Early Stopping         │
│               TensorBoard Logging · GPU (CUDA) Accel             │
└──────────────────────────────────────────────────────────────────┘
```

---

##  1. Model Design and Implementation

All models use **Stable-Baselines3 (SB3)** with `MlpPolicy` backed by **PyTorch 2.x** and automatic GPU/CUDA acceleration. The shared observation and action space design is as follows:

### Shared Observation Space — 74 Dimensions

| Component | Dimensions | Range | Description |
|:---|:---:|:---:|:---|
| Frame-Stacked LiDAR | 72 (24 × 3) | `[0, 1]` | 3 consecutive 24-bin min-pooled LiDAR scans, normalised by `max_range = 3.5 m` |
| Normalised Goal Distance | 1 | `[0, 1]` | Euclidean distance to goal / `max_goal_distance (10.0 m)` |
| Normalised Goal Heading | 1 | `[0, 1]` | Relative angle to goal, shifted from `[-1,1]` → `[0,1]` |

### Action Spaces

| Algorithm | Type | Shape | Mapping |
|:---|:---:|:---:|:---|
| DQN | `Discrete(8)` | Scalar | 8 predefined `(v, ω)` pairs including reverse + turning |
| DDPG, TD3, SAC | `Box(2)` | `[-1, 1]²` | `action[0]` → linear vel `[-v_max, +v_max]`, `action[1]` → angular vel `[-ω_max, +ω_max]` |

> **Motor Safety:** All velocity commands are clamped via differential-drive kinematics to ensure wheel speeds never exceed `6.0 rad/s`.

---

### DQN — Deep Q-Network

| Property | Value |
|:---|:---|
| **Algorithm** | Deep Q-Network (discrete actions) |
| **Policy** | `MlpPolicy` |
| **Network Architecture** | `[256, 256]` fully-connected Q-network |
| **Input** | 74D observation → 256 → ReLU → 256 → ReLU → 8 Q-values |
| **Output** | 8 discrete actions (5 forward + 3 reverse) |
| **Exploration** | ε-greedy: ε = 1.0 → 0.05 over 30% of training |
| **Target Network Update** | Hard copy every 1,000 steps (τ = 1.0) |
| **Replay Buffer** | 100,000 transitions |
| **Batch Size** | 64 |
| **Learning Rate** | 3 × 10⁻⁴ |
| **Discount Factor (γ)** | 0.99 |
| **Training Frequency** | Every 4 environment steps |
| **Default Timesteps** | 100,000 |

---

### DDPG — Deep Deterministic Policy Gradient

| Property | Value |
|:---|:---|
| **Algorithm** | Deep Deterministic Policy Gradient (continuous) |
| **Policy** | `MlpPolicy` |
| **Actor Network** | `[256, 256]` → 2D continuous action (tanh output) |
| **Critic Network** | `[256, 256]` → single Q-value |
| **Input** | 74D observation + 2D action (critic) |
| **Exploration Noise** | Ornstein-Uhlenbeck (σ = 0.15) — temporally correlated |
| **Target Network Update** | Polyak averaging (τ = 0.005) |
| **Replay Buffer** | 100,000 transitions |
| **Batch Size** | 256 |
| **Learning Rate** | 3 × 10⁻⁴ |
| **Discount Factor (γ)** | 0.99 |
| **Training Frequency** | Every 1 environment step |
| **Default Timesteps** | 50,000 |

---

### TD3 — Twin Delayed DDPG

| Property | Value |
|:---|:---|
| **Algorithm** | Twin Delayed DDPG (continuous) |
| **Policy** | `MlpPolicy` |
| **Actor Network** | `[256, 256]` → 2D continuous action (tanh output) |
| **Twin Critic Networks** | 2 × `[256, 256]` → Q-value each; `min(Q₁, Q₂)` used |
| **Input** | 74D observation + 2D action (critics) |
| **Exploration Noise** | Gaussian (σ = 0.15) |
| **Target Policy Smoothing** | Noise σ = 0.2, clipped at ± 0.5 |
| **Delayed Policy Updates** | Actor updated every 2 critic updates |
| **Target Network Update** | Polyak averaging (τ = 0.005) |
| **Replay Buffer** | 100,000 transitions |
| **Batch Size** | 256 |
| **Learning Rate** | 3 × 10⁻⁴ |
| **Discount Factor (γ)** | 0.99 |
| **Training Frequency** | Every 1 environment step |
| **Default Timesteps** | 100,000 |

**TD3 Key Innovations over DDPG:**
1. **Twin Critics** — Takes minimum of two Q-values to combat overestimation bias  
2. **Delayed Policy Updates** — Actor updates lag behind critic by `policy_delay = 2` steps  
3. **Target Policy Smoothing** — Adds clipped noise to target actions for regularisation  

---

### SAC — Soft Actor-Critic (Modified)

| Property | Value |
|:---|:---|
| **Algorithm** | Soft Actor-Critic with automatic entropy tuning |
| **Policy** | `MlpPolicy` |
| **Actor Network** | `[256, 256]` → Gaussian policy (outputs μ and σ) |
| **Twin Critic Networks** | 2 × `[256, 256]` → Q-value each; `min(Q₁, Q₂)` used |
| **Input** | 74D observation |
| **Entropy Coefficient** | `"auto"` — learned via dual-gradient descent |
| **Target Entropy** | `"auto"` = `−dim(action_space)` = `−2` |
| **State-Dependent Exploration** | Optional (SDE) for smoother continuous actions |
| **Target Network Update** | Polyak averaging (τ = 0.005) |
| **Replay Buffer** | 100,000 transitions |
| **Batch Size** | 256 |
| **Learning Rate** | 3 × 10⁻⁴ |
| **Discount Factor (γ)** | 0.99 |
| **Training Frequency** | 1 gradient step per environment step |
| **Default Timesteps** | 50,000 |

**SAC Modifications for Navigation:**
1. **Automatic entropy tuning** — learns the optimal exploration-exploitation balance  
2. **Navigation-aware dense reward shaping** — phase-specific reward coefficients  
3. **Frame-stacked observations** — 3-frame LiDAR for temporal inference about dynamic obstacles  

---

##  2. Algorithmic Behavior and Phase Policies

### Curriculum Learning: 4-Phase Training Pipeline

The training follows a **progressive curriculum** where the agent's weights are **preserved across phases** but the environment complexity increases. Reward coefficients automatically adapt per phase to guide learning focus.

```
Phase 1             Phase 2             Phase 3              Phase 4
┌──────────┐       ┌──────────┐       ┌──────────┐        ┌──────────┐
│  Empty   │  ───► │  Static  │  ───► │   Slow   │  ───►  │   Fast   │
│  Room    │       │ Obstacles│       │ Dynamic  │        │ Dynamic  │
│          │       │ (4 boxes)│       │(2 cyls,  │        │(1 box +  │
│ Learn:   │       │          │       │ 0.06 m/s)│        │ 2 cyls,  │
│ Goal-    │       │ Learn:   │       │          │        │ 0.15 m/s)│
│ seeking  │       │ Obstacle │       │ Learn:   │        │          │
│          │       │ avoidance│       │ Velocity │        │ Learn:   │
│          │       │          │       │ prediction│       │ Agility  │
└──────────┘       └──────────┘       └──────────┘        └──────────┘
 Weights preserved ──────────────────────────────────────────────►
```

#### Phase-Specific Environment Configuration

| Phase | Label | Static Obstacles | Dynamic Obstacles | Dynamic Speed | Max Episode Steps |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | Empty Room | 0 (hidden) | 0 (hidden) | — | 200 |
| 2 | Static Obstacles | 4 boxes (0.25×0.25×0.3 m) | 0 (hidden) | — | 200 |
| 3 | Slow Dynamic | 0 (hidden) | 2 cylinders (r=0.10 m) | 0.06 m/s | 350 |
| 4 | Fast Dynamic | 1 box | 2 cylinders (r=0.10 m) | 0.15 m/s | 300 |

#### Phase-Specific Timestep Budgets

| Phase | DQN | DDPG | TD3 | SAC |
|:---:|:---:|:---:|:---:|:---:|
| 1 — Empty Room | 30,000 | 25,000 | 20,000 | 20,000 |
| 2 — Static Obstacles | 150,000 | 150,000 | 100,000 | 100,000 |
| 3 — Slow Dynamic | 150,000 | 100,000 | 200,000 | 200,000 |
| 4 — Fast Dynamic | 250,000 | 200,000 | 200,000 | 200,000 |

---

### Per-Algorithm Behavioral Analysis

#### 1. DQN — Structured Exploration with Discrete Manoeuvres

| Phase | Policy Behavior |
|:---|:---|
| **Phase 1 (Empty Room)** | ε-greedy exploration (ε: 1.0→0.05) rapidly discovers goal-seeking. 8 discrete actions include dedicated "straight" (max speed) and "sharp turn" (slow + max rotation) primitives. Agent learns to point-and-shoot toward goals. |
| **Phase 2 (Static Obstacles)** | Relies on the 3 reverse actions (backward, reverse-left, reverse-right) to escape near-collision situations. Discrete action space limits smooth obstacle-hugging — tends to make conservative wide detours. |
| **Phase 3 (Slow Dynamic)** | Frame-stacked LiDAR allows detecting obstacle motion across 3 frames. DQN selects from fixed velocity pairs — limited ability to fine-tune avoidance trajectories. Often resorts to "stop-and-wait" via reverse actions. |
| **Phase 4 (Fast Dynamic)** | Struggles most here due to discrete action space — cannot produce the smooth continuous adjustments needed to thread between fast obstacles and a static box. High timeout rate expected. |

#### 2. DDPG — Continuous Control with Correlated Exploration

| Phase | Policy Behavior |
|:---|:---|
| **Phase 1 (Empty Room)** | Ornstein-Uhlenbeck (OU) noise provides temporally correlated exploration — produces smooth, physically realistic trajectories. Learns efficient straight-line-to-goal policies. Single-critic architecture can overestimate Q-values. |
| **Phase 2 (Static Obstacles)** | OU noise helps explore continuous avoidance manoeuvres (smooth curves around boxes). However, single critic leads to overestimation bias — agent may develop overly aggressive approach velocities. |
| **Phase 3 (Slow Dynamic)** | Continuous actions allow fine-grained velocity modulation near moving obstacles. OU noise temporal correlation can be detrimental — agent may commit to avoiding in one direction too long. |
| **Phase 4 (Fast Dynamic)** | Most fragile in this phase. Single critic + deterministic policy mean the agent can get stuck in local optima. Overestimation may cause confident-but-wrong obstacle approach. Higher collision rate expected. |

#### 3. TD3 — Stable Continuous Control with Conservative Estimation

| Phase | Policy Behavior |
|:---|:---|
| **Phase 1 (Empty Room)** | Gaussian noise exploration (σ=0.15) with twin critics preventing overestimation. Delayed policy updates yield more stable initial learning. Learns robust goal-seeking with less variance than DDPG. |
| **Phase 2 (Static Obstacles)** | Conservative Q-estimation (min of twin critics) prevents reckless approach toward obstacles. Target policy smoothing acts as implicit regularisation — produces smoother avoidance trajectories. |
| **Phase 3 (Slow Dynamic)** | Twin critics provide stable value estimation even as the environment dynamics shift. Delayed actor updates mean the policy changes less frequently — more consistent avoidance behavior under non-stationary obstacle positions. |
| **Phase 4 (Fast Dynamic)** | Best-suited for this phase. Conservative estimation + smoothing produce careful, adaptive navigation. Policy changes are deliberate (every 2nd critic update), reducing oscillatory behavior near fast obstacles. |

#### 4. SAC (Modified) — Maximum-Entropy Exploration with Automatic Tuning

| Phase | Policy Behavior |
|:---|:---|
| **Phase 1 (Empty Room)** | Maximum entropy framework drives exploration by maximising both reward and action entropy. Auto-tuned entropy coefficient starts high (exploratory) → decreases as goal-seeking is learned. Stochastic policy naturally covers the action space. |
| **Phase 2 (Static Obstacles)** | Stochastic policy handles multi-modal action distributions (e.g., "go left OR right around a box"). Entropy bonus encourages discovering multiple avoidance strategies rather than committing to a single path. |
| **Phase 3 (Slow Dynamic)** | Entropy-regularised policy remains adaptive to changing obstacle configurations. Automatic entropy tuning adjusts exploration pressure based on task difficulty — increases when facing novel dynamic patterns. |
| **Phase 4 (Fast Dynamic)** | Strong candidate for best performance. Stochastic policy + entropy bonus provide natural exploration-exploitation balance. Twin critics (inherited from SAC's architecture, same as TD3) prevent overestimation. Sample-efficient off-policy learning maximises use of limited Phase 4 data. |

---

##  3. Reward Structure

The reward function is an **8-component dense reward signal** designed to provide rich learning signal at every step. Two components are **terminal** (end the episode), and six are **shaping rewards** applied per step.

### Terminal Rewards

| Reward Component | Value | Condition |
|:---|:---:|:---|
| **R_goal** | **+250.0** | Goal reached (distance to goal ≤ `0.35 m`) |
| **R_collision** | **−300.0** | Collision detected (min LiDAR reading ≤ `0.28 m`) |

### Shaping Rewards (Per-Step)

| Reward Component | Formula | Description |
|:---|:---|:---|
| **R_progress** | `coeff × (prev_dist − curr_dist)` | Reward for getting closer to goal; **clamped to `[−20, +20]`** |
| **R_step** | `coeff` (negative) | Time penalty applied every step to encourage efficiency |
| **R_heading** | `coeff × cos(θ_relative)` | Bonus for facing the goal; penalises turning away |
| **R_angular** | `coeff × \|ω / ω_max\|` | Penalty for excessive rotation (anti-spinning) |
| **R_proximity** | `coeff × 0.1` | Penalty when nearest obstacle is in danger zone: `0.28 m < d < 0.35 m` |
| **R_stuck** | `coeff × 0.1` | Penalty when robot moves < `0.10 m` over the last 50 steps |

### Phase-Specific Reward Coefficients

The reward coefficients **automatically adapt** based on the `CURRICULUM_PHASE` environment variable, shifting the learning focus as complexity increases:

| Coefficient | Phase 1 (Empty) | Phase 2 (Static) | Phase 3 (Slow Dynamic) | Phase 4 (Fast Dynamic) |
|:---|:---:|:---:|:---:|:---:|
| **R_heading** | `+1.5` | `+0.3` | `+0.2` | `+0.1` |
| **R_angular** | `−0.5` | `−0.05` | `−0.03` | `−0.02` |
| **R_proximity** | `−3.0` | `−4.0` | `−6.0` | `−8.0` |
| **R_stuck** | `−5.0` | `−5.0` | `−1.0` | `−1.0` |
| **R_progress** | `200.0` | `200.0` | `150.0` | `100.0` |
| **R_step** | `−2.0` | `−2.0` | `−1.5` | `−1.5` |

#### Design Rationale by Phase

| Phase | Emphasis | Why |
|:---|:---|:---|
| **Phase 1** | Strong heading (`1.5`) + high progress (`200`) + harsh anti-spin (`−0.5`) | Forces the agent to learn direct goal-seeking without aimless wandering |
| **Phase 2** | Weak heading (`0.3`) + strong proximity (`−4.0`) + minimal angular penalty (`−0.05`) | Allows evasive curved manoeuvres around static boxes while still penalising proximity |
| **Phase 3** | Weakest heading (`0.2`) + massive proximity (`−6.0`) + low progress (`150`) + soft step (`−1.5`) | Encourages **patience near moving obstacles** — detours and waiting are acceptable |
| **Phase 4** | Minimal heading (`0.1`) + strongest proximity (`−8.0`) + lowest progress (`100`) + soft step (`−1.5`) | Prioritises **survival over speed** — agent must respect fast obstacles above all else |

### Total Reward Formula

```
R_total = R_goal (+250) + R_collision (−300) + R_progress (coeff × Δd)
        + R_step (coeff) + R_heading (coeff × cos θ) + R_angular (coeff × |ω/ω_max|)
        + R_proximity (coeff × 0.1) + R_stuck (coeff × 0.1)
```

> **Safety:** If any reward computation produces `NaN` or `±Inf`, it is replaced with `−50.0`. All observations are also sanitised to prevent NaN propagation into neural network weights.

---

##  4. Performance Metrics

All evaluation results below are from **200-episode deterministic evaluation runs** on Phase 4 (Fast Dynamic Obstacles) — the hardest environment configuration. Training convergence data is extracted from saved training metadata across all completed curriculum phases.

### Metrics Tracked

| Metric | Description |
|:---|:---|
| **Success Rate** | Fraction of episodes where the robot reached the goal (`distance ≤ 0.35 m`) |
| **Collision Rate** | Fraction of episodes ending in collision (`min LiDAR ≤ 0.28 m`) |
| **Timeout Rate** | Fraction of episodes exceeding max step limit without goal or collision |
| **Avg. Cumulative Reward** | Mean total reward per episode |
| **Avg. Episode Length** | Mean number of steps per episode |
| **Avg. Final Goal Distance** | Mean distance to goal at episode termination |
| **Training Time** | Wall-clock duration per phase |
| **Steps/Second** | Training throughput |

---

###  Final Model Performance Comparison — Phase 4, 200 Episodes

| Metric | 🥇 SAC | 🥈 TD3 | 🥉 DQN | DDPG |
|:---|:---:|:---:|:---:|:---:|
| **Success Rate** | **61.0%** | 58.0% | 57.0% | 51.0% |
| **Collision Rate** | **40.0%** | 44.5% | 45.5% | 49.0% |
| **Timeout Rate** | ~0% | ~0% | ~0% | 2.0% |
| **Avg. Reward** | **+40.3** | +11.8 | +18.4 | −53.8 |
| **Avg. Episode Length** | 19 steps | 24 steps | 19 steps | 46 steps |
| **Avg. Final Goal Dist.** | **0.59 m** | 0.66 m | 0.67 m | 0.69 m |

> **Winner: Modified SAC** — highest success rate (61.0%), lowest collision rate (40.0%), highest average reward (+40.3), and shortest average goal distance at termination (0.59 m).

---

### Training Convergence — Per-Phase Breakdown

| Algorithm | Phase | Label | Timesteps | Wall-Clock Time |
|:---|:---:|:---|---:|---:|
| **SAC** | 1 | Empty Room | 25,000 | 0:10:35 |
| **SAC** | 2 | Static Obstacles | 100,000 | 0:53:26 |
| **SAC** | 3 | Slow Dynamic | 200,000 | 0:34:14 |
| **SAC** | **Total** | **Phases 1–3** | **325,000** | **1:38:15** |
| | | | | |
| **TD3** | 1 | Empty Room | 100,000 | 0:33:02 |
| **TD3** | 2 | Static Obstacles | 100,000 | 2:10:50 |
| **TD3** | 3 | Slow Dynamic | 200,000 | 0:08:44 |
| **TD3** | 4 | Fast Dynamic | 50,000 | 0:50:20 |
| **TD3** | **Total** | **Phases 1–4** | **450,000** | **3:42:58** |
| | | | | |
| **DDPG** | 1 | Empty Room | 25,000 | 0:20:38 |
| **DDPG** | **Total** | **Phase 1 only** | **25,000** | **0:20:38** |
| | | | | |
| **DQN** | 4 | Fast Dynamic | 250,000 | 4:28:45 |
| **DQN** | **Total** | **Phase 4 only** | **250,000** | **4:28:45** |

---

### Convergence Speed Ranking

| Rank | Algorithm | Total Timesteps | Total Wall-Clock Time | Avg. Steps/sec |
|:---:|:---|---:|---:|---:|

| 1 | **SAC** | 120,000 | 1:38:15 | 34.1 |
| 2 | **TD3** | 230,000 | 3:42:58 | 34.6 |
| 3 | **DDPG** | 305,000 | 4:20:38 | 55.2 |
| 4 | **DQN** | 390,000 | 4:28:45 | 22 |


---

###  Key Observations & Analysis

#### 1. Spawn-on-Goal Bias (All Algorithms)

A significant fraction of episodes terminate at `len=1` with `reward ≈ +250.0`, indicating the robot spawned within the `0.35 m` goal threshold. This inflates success rates across all algorithms:

| Algorithm | Instant Successes (len=1, GOAL) | Instant Crashes (len=1, CRASH) | Net Spawn Bias |
|:---|:---:|:---:|:---|
| SAC | ~28 / 200 (14%) | ~15 / 200 (7.5%) | +6.5% net success inflation |
| TD3 | ~22 / 200 (11%) | ~16 / 200 (8%) | +3% net success inflation |
| DQN | ~16 / 200 (8%) | ~14 / 200 (7%) | +1% net success inflation |
| DDPG | ~14 / 200 (7%) | ~12 / 200 (6%) | +1% net success inflation |

> **Interpretation:** The random spawn + random goal placement occasionally creates trivially solvable or impossible episodes. This is a simulation artefact that affects all algorithms equally but should be considered when interpreting absolute success rates.

#### 2. DDPG Degradation — Single-Critic Fragility

DDPG shows uniquely severe failure modes not observed in the other algorithms:

- **Catastrophic episodes:** DDPG produces rewards as extreme as `−524.6` (Ep 74, len=171) and `−513.5` (Ep 51, len=191) — far worse than any SAC/TD3/DQN episode. These indicate the agent driving in extended collision loops without recovering.
- **Longest average episodes (46 steps)** — 2.4× longer than SAC/DQN (19 steps), suggesting the agent wanders without efficient goal-seeking.
- **Only algorithm with timeouts:** 4 episodes (Eps 13, 24, 33, 142) hit the 300-step limit — the agent gets trapped in orbit patterns around obstacles.
- **Phase 1–only training:** DDPG completed only Phase 1 (empty room) yet was evaluated on Phase 4 (fast dynamic obstacles). The absence of obstacle avoidance training directly explains its 49% collision rate.

#### 3. DQN's Surprising Competitiveness

Despite using a **discrete action space** (8 fixed velocity pairs) versus the continuous action spaces of DDPG/TD3/SAC, DQN achieves **57% success** — only 1% behind TD3 and 4% behind SAC:

- **Direct-to-Phase-4 training:** DQN was trained only on Phase 4 (250K steps), skipping the curriculum entirely — yet achieves competitive performance.
- **Discrete advantage in dynamic environments:** The 8-action lookup table includes dedicated reverse manoeuvres that may provide more decisive escape behaviour than continuous policies that output small, uncertain velocities near obstacles.
- **Matched episode efficiency:** DQN's average episode length (19 steps) matches SAC, suggesting its discrete actions don't waste time with hesitant partial movements.

#### 4. SAC's Maximum-Entropy Advantage

SAC's dominance is attributable to specific architectural advantages:

- **Automatic entropy tuning** dynamically adjusts exploration pressure — high entropy in novel Phase 4 obstacle configurations, low entropy on familiar goal approaches.
- **Stochastic policy** naturally handles the multi-modality of obstacle avoidance (go-left vs. go-right around a cylinder) — deterministic policies (DDPG, TD3) must commit to one direction.
- **Best sample efficiency:** SAC achieves the top success rate with only 325K total training steps (vs. TD3's 450K), and the highest training throughput at 55.1 steps/sec.
- **Tightest reward distribution:** SAC's successful episodes cluster narrowly around `+250` to `+310`, with fewer extreme negative outliers than TD3 or DDPG.

#### 5. TD3 vs. SAC: The Conservative Estimation Trade-off

TD3 and SAC are architecturally similar (twin critics, off-policy), but differ in their exploration strategy:

| Aspect | SAC | TD3 |
|:---|:---|:---|
| Exploration | Entropy-driven (adaptive) | Fixed Gaussian noise (σ=0.15) |
| Policy type | Stochastic | Deterministic + noise |
| Q-value estimation | Min of twin critics | Min of twin critics + target smoothing |
| Phase 4 success | **61.0%** | 58.0% |
| Phase 4 collisions | **40.0%** | 44.5% |
| Avg. episode length | **19 steps** | 24 steps |

TD3's slightly longer episodes (24 vs. 19) suggest more cautious navigation — the target policy smoothing creates overly conservative action selection near obstacles, causing the agent to slow down and eventually get clipped by fast-moving cylinders.

#### 6. Curriculum Completeness vs. Direct Training

A key experimental finding is the varying curriculum coverage across algorithms:

| Algorithm | Phases Completed | Phase 4 Success |
|:---|:---|:---:|
| SAC | 1 → 2 → 3 (no P4 training) | **61.0%** |
| TD3 | 1 → 2 → 3 → 4 (full) | 58.0% |
| DQN | 4 only (direct) | 57.0% |
| DDPG | 1 only | 51.0% |

> **Remarkable finding:** SAC achieves the highest Phase 4 performance **without ever being explicitly trained on Phase 4**. Its Phase 3 (slow dynamic) training transferred effectively to the faster Phase 4 obstacles — evidence that SAC's entropy-regularised policy learns generalisable avoidance behaviours rather than speed-specific ones.

---

##  Environment Specification

### Simulation Arena

| Parameter | Value |
|:---|:---|
| Simulator | Webots R2025a |
| Arena Dimensions | 3.0 m × 3.0 m (walls at ±1.5 m) |
| Floor | PBR grid-textured plane |
| Walls | 4 solid walls (3.1 m × 0.1 m × 0.4 m), grey PBR appearance |
| Time Step | 16 ms |
| Physics | ODE engine with soft constraints (CFM=0.001, ERP=0.2) |

### Robot — TurtleBot3 Burger

| Parameter | Value |
|:---|:---|
| Platform | TurtleBot3 Burger (ROBOTIS) |
| Drive | Differential drive |
| Max Linear Velocity | 0.22 m/s |
| Max Angular Velocity | 2.0 rad/s |
| Wheel Radius | 0.033 m |
| Track Width | 0.160 m |
| Max Wheel Velocity | 6.0 rad/s |
| LiDAR | 360° laser scanner, max range 3.5 m |

### Obstacle Details

| Object | Type | Geometry | Appearance |
|:---|:---|:---|:---|
| Static Boxes | `STATIC_1` – `STATIC_4` | Box 0.25 × 0.25 × 0.3 m | Wooden brown PBR |
| Dynamic Cylinders | `OBS_1` – `OBS_3` | Cylinder r=0.10 m, h=0.20 m | Red, Blue, Green PBR |
| Goal Marker | `GOAL_MARKER` | Cylinder r=0.08 m, h=0.02 m | Glowing green (emissive) |
| Start Marker | `START_MARKER` | Cylinder r=0.06 m, h=0.02 m | Glowing blue (emissive) |

### Dynamic Obstacle Trajectories (Phases 3 & 4)

| Trajectory Type | Description | Parameters |
|:---|:---|:---|
| **Linear** | Straight line between two random wall edges | Start/end on opposite edges; ping-pong at endpoints |
| **Parabolic** | Curved arc with perpendicular offset | Arc height: `0.4–0.8 m` (random sign); offset = `h × 4t(1−t)` |

> Trajectory pairs are validated at generation time to ensure minimum `0.7 m` separation in Phase 3 and `0.5 m` separation in Phase 4 (including from static boxes).

---

##  Installation & Usage

### Prerequisites

- **Ubuntu 22.04** (WSL2 supported)
- **ROS2 Humble**
- **Webots R2025a**
- **Python 3.10+**
- **NVIDIA GPU with CUDA** (recommended, CPU fallback supported)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/VaibhavDeopa/Autonomous-Robot-Navigation-in-Dynamic-Environment.git H2F
cd H2F

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Deploy ROS2 workspace
bash deploy_to_ros2_ws.sh

# 4. Verify GPU availability
python3 gpu_check.py
```

### Curriculum Training

```bash
# ── Phase 1: Empty Room ───────────────────────────────
export CURRICULUM_PHASE=1
ros2 launch nav_env_pkg nav_launch.py   # Terminal 1: start Webots
python3 train_curriculum.py --algo sac --phase 1 --timesteps 20000   # Terminal 2

# ── Phase 2: Static Obstacles ─────────────────────────
# Stop Webots (Ctrl+C), then:
export CURRICULUM_PHASE=2
ros2 launch nav_env_pkg nav_launch.py
python3 train_curriculum.py --algo sac --phase 2 --timesteps 100000

# ── Phase 3: Slow Dynamic ────────────────────────────
export CURRICULUM_PHASE=3
ros2 launch nav_env_pkg nav_launch.py
python3 train_curriculum.py --algo sac --phase 3 --timesteps 200000

# ── Phase 4: Fast Dynamic ────────────────────────────
export CURRICULUM_PHASE=4
ros2 launch nav_env_pkg nav_launch.py
python3 train_curriculum.py --algo sac --phase 4 --timesteps 200000
```

### Evaluation

```bash
# Compare training metrics only (no simulator needed)
python3 evaluate_curriculum.py --metrics-only

# Full evaluation with live simulation (50 episodes per algorithm)
python3 evaluate_curriculum.py --episodes 50

# TensorBoard visualization
tensorboard --logdir logs/
```

---

##  Project Structure

```
H2F/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── config.py                    # Central hyperparameters & experiment configuration
├── nav_env.py                   # Gymnasium environment (ROS2 ↔ RL bridge)
├── utils.py                     # Callbacks, metrics, checkpointing, training timer
│
├── train_dqn.py                 # DQN standalone training script
├── train_ddpg.py                # DDPG standalone training script
├── train_td3.py                 # TD3 standalone training script
├── train_sac.py                 # Modified SAC standalone training script
├── train_all.py                 # Run all 4 algorithms sequentially
├── train_curriculum.py          # 4-phase curriculum training pipeline
│
├── evaluate.py                  # Single-phase model evaluation
├── evaluate_curriculum.py       # Cross-algorithm, cross-phase comparison
│
├── supervisor_plugin.py         # Webots supervisor: obstacle management, resets
├── supervisor_controller.py     # Supervisor ROS2 controller node
│
├── square_arena.wbt             # Webots world file (3×3m arena)
├── turtlebot3_burger.urdf       # Robot URDF description
├── supervisor.urdf              # Supervisor URDF
├── ros2control.yml              # ROS2 control configuration
│
├── nav_launch.py                # ROS2 launch file
├── navigation_sim.launch.py     # Alternative launch configuration
├── deploy_to_ros2_ws.sh         # Deploys project to ROS2 workspace
├── setup_wsl.sh                 # WSL2 environment setup script
├── setup.py                     # Python package setup
│
├── gpu_check.py                 # CUDA/GPU verification utility
├── check_webots.py              # Webots installation checker
└── check_supervisor.sh          # Supervisor health check
```

---

##  License

This project was developed as part of a H2F hackathon at IIIT Dharwad (Vocab.AI) on Deep Reinforcement Learning for Autonomous Navigation. 

---

<p align="center">
</p>

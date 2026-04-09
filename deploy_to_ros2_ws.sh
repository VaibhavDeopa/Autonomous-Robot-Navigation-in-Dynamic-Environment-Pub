#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# deploy_to_ros2_ws.sh — Deploy H2F files into the ROS2 workspace
# ═══════════════════════════════════════════════════════════════════════════════
# Usage: bash deploy_to_ros2_ws.sh
#
# This script copies all project files into the correct locations within
# the nav_env_pkg ROS2 package at ~/ros2_ws/src/nav_env_pkg/
# Supports the 4-phase curriculum training pipeline.

set -e

ROS2_WS="$HOME/ros2_ws"
PKG_DIR="$ROS2_WS/src/nav_env_pkg"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "═══════════════════════════════════════════════════════════════"
echo "  Deploying H2F Navigation Project to ROS2 Workspace"
echo "═══════════════════════════════════════════════════════════════"
echo "  Source:      $SCRIPT_DIR"
echo "  Destination: $PKG_DIR"
echo ""

# ── Create directory structure ────────────────────────────────────────
echo "📁  Creating directory structure..."
mkdir -p "$PKG_DIR/nav_env_pkg"
mkdir -p "$PKG_DIR/worlds"
mkdir -p "$PKG_DIR/launch"
mkdir -p "$PKG_DIR/resource"
mkdir -p "$PKG_DIR/controllers/supervisor_controller"
mkdir -p "$PKG_DIR/scripts"

# ── Copy Python modules ──────────────────────────────────────────────
echo "🐍  Copying Python modules..."
cp "$SCRIPT_DIR/nav_env.py" "$PKG_DIR/nav_env_pkg/nav_env.py"
cp "$SCRIPT_DIR/supervisor_plugin.py" "$PKG_DIR/nav_env_pkg/supervisor_plugin.py"

# Ensure __init__.py exists
touch "$PKG_DIR/nav_env_pkg/__init__.py"

# ── Copy Supervisor Controller ────────────────────────────────────────
echo "🎮  Copying supervisor controller..."
if [ -f "$SCRIPT_DIR/supervisor_controller.py" ]; then
    cp "$SCRIPT_DIR/supervisor_controller.py" \
       "$PKG_DIR/controllers/supervisor_controller/supervisor_controller.py"
fi

# ── Copy World Files ─────────────────────────────────────────────────
echo "🌍  Copying Webots world files..."
if [ -f "$SCRIPT_DIR/turtlebot3_arena.wbt" ]; then
    cp "$SCRIPT_DIR/turtlebot3_arena.wbt" "$PKG_DIR/worlds/turtlebot3_arena.wbt"
fi
cp "$SCRIPT_DIR/square_arena.wbt" "$PKG_DIR/worlds/square_arena.wbt"

# ── Copy Resource Files (URDF, ros2control) ───────────────────────────
echo "📋  Copying resource files..."
cp "$SCRIPT_DIR/turtlebot3_burger.urdf" "$PKG_DIR/resource/turtlebot3_burger.urdf"
cp "$SCRIPT_DIR/supervisor.urdf" "$PKG_DIR/resource/supervisor.urdf"
cp "$SCRIPT_DIR/ros2control.yml" "$PKG_DIR/resource/ros2control.yml"

# ── Copy Launch Files ────────────────────────────────────────────────
echo "🚀  Copying launch files..."
cp "$SCRIPT_DIR/nav_launch.py" "$PKG_DIR/launch/nav_launch.py"
if [ -f "$SCRIPT_DIR/navigation_sim.launch.py" ]; then
    cp "$SCRIPT_DIR/navigation_sim.launch.py" \
       "$PKG_DIR/launch/navigation_sim.launch.py"
fi

# ── Copy Training Scripts (to scripts/ for reference, also to pkg root) ──
echo "📊  Copying training scripts..."
for f in config.py utils.py train_curriculum.py train_td3.py train_ddpg.py \
         train_dqn.py train_sac.py train_all.py evaluate.py evaluate_curriculum.py; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_DIR/$f" "$PKG_DIR/scripts/$f"
    fi
done

# ── Update package.xml ───────────────────────────────────────────────
echo "📦  Updating package.xml..."
cat > "$PKG_DIR/package.xml" << 'PACKAGE_XML'
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>nav_env_pkg</name>
  <version>1.0.0</version>
  <description>DRL Navigation Environment for TurtleBot3 in Webots</description>
  <maintainer email="uday@todo.todo">uday</maintainer>
  <license>MIT</license>

  <depend>rclpy</depend>
  <depend>webots_ros2_driver</depend>
  <depend>sensor_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>std_msgs</depend>
  <depend>std_srvs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
PACKAGE_XML

# ── Update setup.py ──────────────────────────────────────────────────
echo "⚙️  Updating setup.py..."
cat > "$PKG_DIR/setup.py" << 'SETUP_PY'
import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'nav_env_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install world files
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.wbt')),
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        # Install resource files (URDF, ros2control)
        (os.path.join('share', package_name, 'resource'),
            glob('resource/*.urdf') + glob('resource/*.yml')),
        # Install controller files
        (os.path.join('share', package_name, 'controllers', 'supervisor_controller'),
            glob('controllers/supervisor_controller/*.py')),
        # Install training scripts
        (os.path.join('share', package_name, 'scripts'),
            glob('scripts/*.py')),
    ],
    install_requires=[
        'setuptools',
        'gymnasium',
        'numpy',
    ],
    zip_safe=True,
    maintainer='uday',
    maintainer_email='uday@todo.todo',
    description='DRL Navigation Environment for TurtleBot3 in Webots',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'nav_env_test = nav_env_pkg.nav_env:main',
        ],
    },
)
SETUP_PY

# ── Update setup.cfg ─────────────────────────────────────────────────
cat > "$PKG_DIR/setup.cfg" << 'SETUP_CFG'
[develop]
script_dir=$base/lib/nav_env_pkg
[install]
install_scripts=$base/lib/nav_env_pkg
SETUP_CFG

# ── Ensure resource marker exists ────────────────────────────────────
touch "$PKG_DIR/resource/nav_env_pkg"

# ── Build ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Building the workspace..."
echo "═══════════════════════════════════════════════════════════════"
cd "$ROS2_WS"

# Source ROS2
source /opt/ros/humble/setup.bash 2>/dev/null || true

# Build only our package
colcon build --packages-select nav_env_pkg --symlink-install
source install/setup.bash

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅  Deployment complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Package structure:"
find "$PKG_DIR" -type f \( -name "*.py" -o -name "*.wbt" -o -name "*.xml" -o -name "*.urdf" -o -name "*.yml" \) | sort | sed 's|'"$ROS2_WS"'|  ~/ros2_ws|'
echo ""
echo "  Next steps (Phase 2 Training):"
echo "    Terminal 1 (Webots):"
echo "      export CURRICULUM_PHASE=2"
echo "      source ~/ros2_ws/install/setup.bash"
echo "      ros2 launch nav_env_pkg nav_launch.py"
echo ""
echo "    Terminal 2 (Training):"
echo "      export CURRICULUM_PHASE=2"
echo "      cd /mnt/c/Users/udayd/Downloads/H2F"
echo "      python3 train_curriculum.py --algo td3 --phase 2"
echo ""

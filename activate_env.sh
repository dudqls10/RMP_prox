#!/bin/bash
# Activate RB10 RMPflow RViz environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Source ROS 2
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Source workspace if built
if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    source "$WORKSPACE_DIR/install/setup.bash"
fi

# Add rmp2 to PYTHONPATH (backup)
export PYTHONPATH="$WORKSPACE_DIR/rmp2:$PYTHONPATH"

echo "Environment activated!"
echo "  - ROS 2: $ROS_DISTRO"
echo "  - Python: $(python3 --version)"
echo "  - TensorFlow: $(python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'not found')"

#!/bin/bash
# Setup virtual environment for RB10 RMPflow RViz

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/venv"

echo "========================================"
echo "Setting up RB10 RMPflow RViz Environment"
echo "========================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "System Python version: $PYTHON_VERSION"

# Create virtual environment with system site packages (for ROS 2)
echo ""
echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR" --system-site-packages

# Activate venv
source "$VENV_DIR/bin/activate"

echo "Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install TensorFlow (CPU version for compatibility)
echo ""
echo "Installing TensorFlow (this may take a while)..."
pip install tensorflow

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install numpy scipy pyyaml urdf-parser-py

# Install rmp2 as editable package
echo ""
echo "Installing rmp2 package..."
cd "$WORKSPACE_DIR/rmp2"
pip install -e .

# Create activation script
echo ""
echo "Creating activation script..."
cat > "$SCRIPT_DIR/activate_env.sh" << 'ACTIVATE_EOF'
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
ACTIVATE_EOF

chmod +x "$SCRIPT_DIR/activate_env.sh"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source $SCRIPT_DIR/activate_env.sh"
echo ""
echo "To build and run:"
echo "  source $SCRIPT_DIR/activate_env.sh"
echo "  cd $WORKSPACE_DIR"
echo "  colcon build --packages-select rb10_rmpflow_rviz"
echo "  source install/setup.bash"
echo "  ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py"
echo ""

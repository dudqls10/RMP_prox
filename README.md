# rb10_rmpflow_rviz

`rb10_rmpflow.launch.py`를 실행해서 RB10 RMP controller와 RViz를 함께 띄우는 ROS 2 Humble 패키지입니다.

현재 구현은 아래 조합을 사용합니다.

- `Eigen`: 수치 계산
- `Pinocchio`: RB10 kinematics / Jacobian
- `CasADi`: task-map / DAG 미분

이 문서는 처음 받은 사람이 GitHub에서 내려받아 바로 실행하는 기준으로 다시 정리한 사용 가이드입니다.

## 1. 권장 환경

- Ubuntu `22.04`
- ROS 2 `Humble`
- 작업공간 경로: `~/rb10_rl_sim`

중요:
- 현재 [`CMakeLists.txt`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/CMakeLists.txt) 는 CasADi 설치 경로를 `~/rb10_rl_sim/third_party/casadi/install` 로 가정합니다.
- 따라서 이 README대로 하면 가장 덜 헷갈립니다.

## 2. 작업공간 받기

`~/rb10_rl_sim` 경로에 작업공간을 받습니다.

```bash
cd ~
git clone <YOUR_GIT_REPO_URL> rb10_rl_sim
cd ~/rb10_rl_sim
```

이미 다른 이름으로 clone 했다면:
- `CMakeLists.txt`의 `CASADI_ROOT`를 수정하거나
- 아래 설치 경로를 직접 맞춰야 합니다.

## 3. 시스템 패키지 설치

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  libeigen3-dev \
  libboost-filesystem-dev \
  libboost-serialization-dev \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-rclcpp \
  ros-humble-std-msgs \
  ros-humble-sensor-msgs \
  ros-humble-geometry-msgs \
  ros-humble-visualization-msgs \
  ros-humble-tf2-ros \
  ros-humble-interactive-markers \
  ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher \
  ros-humble-rviz2 \
  ros-humble-pinocchio
```

`rosdep`를 아직 초기화하지 않았다면:

```bash
sudo rosdep init
rosdep update
```

## 4. CasADi 설치

이 패키지는 `CasADi`를 로컬로 빌드해서 사용합니다.

```bash
mkdir -p ~/rb10_rl_sim/third_party
git clone --depth 1 --branch 3.7.2 https://github.com/casadi/casadi.git ~/rb10_rl_sim/third_party/casadi

cmake -S ~/rb10_rl_sim/third_party/casadi \
  -B ~/rb10_rl_sim/third_party/casadi/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/rb10_rl_sim/third_party/casadi/install \
  -DWITH_PYTHON=OFF \
  -DWITH_PYTHON3=OFF \
  -DWITH_EXAMPLES=OFF

cmake --build ~/rb10_rl_sim/third_party/casadi/build -j"$(nproc)"
cmake --install ~/rb10_rl_sim/third_party/casadi/build
```

정상 설치되면 이 파일이 있어야 합니다.

```bash
ls ~/rb10_rl_sim/third_party/casadi/install/lib/libcasadi.so
```

## 5. 빌드

```bash
source /opt/ros/humble/setup.bash
cd ~/rb10_rl_sim
colcon build --packages-select rb10_rmpflow_rviz --symlink-install
source ~/rb10_rl_sim/install/setup.bash
```

참고:
- 실행 파일에는 CasADi `RPATH`가 들어가 있으므로 보통 `LD_LIBRARY_PATH`를 따로 잡지 않아도 됩니다.

## 6. 실행

### 전체 실행

`robot_state_publisher`, `rmpflow_controller`, `interactive_goal`, `obstacle_manager`, `tof_ray_visualizer`, `rviz2`를 함께 띄웁니다.

```bash
source /opt/ros/humble/setup.bash
source ~/rb10_rl_sim/install/setup.bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py
```

### 옵션

RViz 없이 실행:

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py use_rviz:=false
```

장애물 매니저 없이 실행:

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py use_obstacles:=false
```

모델만 확인:

```bash
ros2 launch rb10_rmpflow_rviz rb10_model_only.launch.py
```

## 7. 실행 후 조작

- RViz에서 goal marker를 움직이면 end-effector 목표가 바뀝니다.
- obstacle manager가 켜져 있으면 obstacle marker를 움직여 회피 동작을 볼 수 있습니다.
- controller는 [`config/params.yaml`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/config/params.yaml) 을 읽어 동작합니다.

## 8. 처음 실행이 안 될 때 확인할 것

### `libcasadi.so`를 못 찾는 경우

아래 파일이 있는지 확인:

```bash
ls ~/rb10_rl_sim/third_party/casadi/install/lib/libcasadi.so
```

만약 작업공간를 `~/rb10_rl_sim`가 아닌 다른 경로에 두었다면:
- [`CMakeLists.txt`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/CMakeLists.txt)의 `CASADI_ROOT`를 수정하고 다시 빌드해야 합니다.

### `pinocchio` 관련 에러가 나는 경우

```bash
sudo apt install -y ros-humble-pinocchio
```

### 빌드는 되는데 launch가 안 되는 경우

아래 순서로 다시 실행:

```bash
source /opt/ros/humble/setup.bash
source ~/rb10_rl_sim/install/setup.bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py
```

## 9. 핵심 파일

- launch: [`rb10_rmpflow.launch.py`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/launch/rb10_rmpflow.launch.py)
- controller 파라미터: [`params.yaml`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/config/params.yaml)
- controller node: [`rmpflow_controller_node.cpp`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/src/rmpflow_controller_node.cpp)
- solver: [`pinocchio_direct_solver.cpp`](/home/song/rb10_rl_sim/rb10_rmpflow_rviz/src/pinocchio_direct_solver.cpp)

## 10. 빠른 실행 요약

```bash
cd ~
git clone <YOUR_GIT_REPO_URL> rb10_rl_sim
cd ~/rb10_rl_sim

sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libeigen3-dev libboost-filesystem-dev libboost-serialization-dev \
  python3-colcon-common-extensions python3-rosdep \
  ros-humble-rclcpp ros-humble-std-msgs ros-humble-sensor-msgs \
  ros-humble-geometry-msgs ros-humble-visualization-msgs \
  ros-humble-tf2-ros ros-humble-interactive-markers \
  ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
  ros-humble-rviz2 ros-humble-pinocchio

mkdir -p ~/rb10_rl_sim/third_party
git clone --depth 1 --branch 3.7.2 https://github.com/casadi/casadi.git ~/rb10_rl_sim/third_party/casadi
cmake -S ~/rb10_rl_sim/third_party/casadi -B ~/rb10_rl_sim/third_party/casadi/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/rb10_rl_sim/third_party/casadi/install \
  -DWITH_PYTHON=OFF -DWITH_PYTHON3=OFF -DWITH_EXAMPLES=OFF
cmake --build ~/rb10_rl_sim/third_party/casadi/build -j"$(nproc)"
cmake --install ~/rb10_rl_sim/third_party/casadi/build

source /opt/ros/humble/setup.bash
colcon build --packages-select rb10_rmpflow_rviz --symlink-install
source ~/rb10_rl_sim/install/setup.bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py
```
# RMP_Proximity-Sensor
# RMP_Proximity-Sensor

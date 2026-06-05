# Local Changes Relative to `f9fac72`

작성일: 2026-05-27

이 문서는 현재 워킹트리에 남아 있는 변경 사항을 기준으로 정리했다. 기준 커밋은 `f9fac72 5/26 compatitable with LPB inference`이다.

## 요약

이번 변경의 중심은 RB10으로 보내는 명령 방식을 `position`과 `velocity`로 선택할 수 있게 만든 것이다.

- `command_mode:=position`: 기존처럼 관절 위치 목표를 `move_servo_j` 또는 joint command topic으로 보낸다.
- `command_mode:=velocity`: RMP 결과의 관절 속도 `qd`를 사용해서 `move_speed_j` 또는 joint command topic으로 보낸다.
- velocity 모드에서는 정지/종료 시 0 속도 명령을 보내도록 보호 로직을 추가했다.
- `rb10_rmpflow.launch.py`와 `rb10_rmpflow_test.launch.py`에 `command_mode` 런치 인자를 추가했다.
- `rb10_api_bridge.py`, `rb10_direct_bridge_node.cpp`, `Rb10SocketClient`가 `move_speed_j`를 지원하도록 확장됐다.
- `params.yaml`에 `speedj_*` 파라미터와 command guard/RMP 튜닝값이 추가 또는 변경됐다.

## 안전 관련 핵심

현재 변경 기준으로 `rb10_rmpflow.launch.py`의 `use_direct_hardware_backend` 기본값이 `"true"`로 되어 있다.

따라서 아래처럼 실행하면 직접 RB10 API 백엔드를 사용할 수 있다.

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py
```

로봇으로 직접 보내지 않고 RViz/토픽 확인 용도로만 쓰려면 명시적으로 끄는 것이 안전하다.

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py \
  use_direct_hardware_backend:=false \
  start_api_bridge:=false
```

`command_mode:=velocity` 자체가 API 직접 송신을 의미하는 것은 아니다. 실제 송신 여부는 `backend_mode`, `use_direct_hardware_backend`, `start_api_bridge` 조합에 따라 달라진다.

## 변경 파일

| 파일 | 주요 변경 |
| --- | --- |
| `config/params.yaml` | `command_mode`, `speedj_*`, RMP/command guard 튜닝 추가 |
| `include/rb10_rmpflow_rviz/rb10_socket_client.hpp` | `send_speedj_degrees_per_sec()` 선언 추가 |
| `src/rb10_socket_client.cpp` | `move_speed_j` 문자열 생성 및 송신 추가 |
| `src/rmpflow_controller_node.cpp` | `command_mode` 파싱, velocity 명령 출력, direct backend speedj 지원 |
| `src/rb10_direct_bridge_node.cpp` | standalone direct bridge에서 velocity 모드 지원 |
| `scripts/rb10_api_bridge.py` | API bridge에서 `move_speed_j` 지원 |
| `scripts/api/cobot.py` | `move_speed_j` 명령도 즉시 송신 허용 |
| `launch/rb10_rmpflow.launch.py` | `command_mode` 런치 인자와 파라미터 전달 추가 |
| `launch/rb10_rmpflow_test.launch.py` | 테스트 런치에도 `command_mode` 전달 추가 |

## 핵심 코드 변경

### 1. command mode 파싱

파일: `src/rmpflow_controller_node.cpp`

```cpp
enum class CommandMode
{
  kPosition,
  kVelocity
};

std::string normalized_command_mode(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return value;
}

CommandMode parse_command_mode(const std::string & value)
{
  const std::string normalized = normalized_command_mode(value);
  if (normalized == "position" || normalized == "servo_j" || normalized == "servoj") {
    return CommandMode::kPosition;
  }
  if (normalized == "velocity" || normalized == "speed_j" || normalized == "speedj") {
    return CommandMode::kVelocity;
  }
  throw std::runtime_error(
    "command_mode must be one of: position, velocity, servo_j, speed_j");
}

const char * command_mode_to_string(CommandMode command_mode)
{
  return command_mode == CommandMode::kVelocity ? "velocity" : "position";
}

const char * rb10_stream_command_name(CommandMode command_mode)
{
  return command_mode == CommandMode::kVelocity ? "move_speed_j" : "move_servo_j";
}
```

의미:

- `position`, `servo_j`, `servoj`는 position 모드로 처리한다.
- `velocity`, `speed_j`, `speedj`는 velocity 모드로 처리한다.
- velocity 모드에서는 RB10 명령 이름이 `move_speed_j`가 된다.

### 2. joint command topic 출력 변경

파일: `src/rmpflow_controller_node.cpp`

```cpp
std_msgs::msg::Float64MultiArray command;
const JointVector & command_vector =
  command_mode_ == CommandMode::kVelocity ? command_state.qd : command_state.q;
command.data.assign(command_vector.data(), command_vector.data() + command_vector.size());

command_pub_->publish(command);
```

의미:

- position 모드에서는 `command_state.q`를 publish한다.
- velocity 모드에서는 `command_state.qd`를 publish한다.
- 즉 같은 topic을 쓰더라도 메시지 내용의 의미가 모드에 따라 달라진다.

### 3. RB10 direct backend에서 speedj 송신

파일: `src/rmpflow_controller_node.cpp`

```cpp
const bool sent =
  command_mode_ == CommandMode::kVelocity ?
  socket_client_.send_speedj_degrees_per_sec(
    joint_deg, speedj_t1_, speedj_t2_, speedj_gain_, speedj_alpha_) :
  socket_client_.send_servoj_degrees(
    joint_deg, servo_t1_, servo_t2_, servo_gain_, servo_alpha_);
```

의미:

- `command_mode:=position`이면 `move_servo_j`를 보낸다.
- `command_mode:=velocity`이면 `move_speed_j`를 보낸다.

### 4. velocity 모드 종료 시 0 속도 송신

파일: `src/rmpflow_controller_node.cpp`

```cpp
~DirectRb10Backend() override
{
  stop_servo_command_thread();
  if (command_mode_ == CommandMode::kVelocity && socket_client_.is_connected()) {
    std::array<double, 6> zero_velocity_deg_s{};
    if (!socket_client_.send_speedj_degrees_per_sec(
        zero_velocity_deg_s, speedj_t1_, speedj_t2_, speedj_gain_, speedj_alpha_))
    {
      RCLCPP_WARN(node_->get_logger(), "Direct RB10 backend failed to send zero move_speed_j on shutdown");
    }
  }
}
```

의미:

- velocity 모드에서 노드가 종료될 때 마지막 속도 명령이 남지 않도록 0 속도 명령을 보낸다.

### 5. speedj 명령 문자열 생성

파일: `src/rb10_socket_client.cpp`

```cpp
bool Rb10SocketClient::send_speedj_degrees_per_sec(
  const std::array<double, 6> & joint_velocity_deg_s,
  double time1,
  double time2,
  double gain,
  double lpf_gain)
{
  std::ostringstream command_stream;
  command_stream << std::fixed << std::setprecision(3);
  command_stream << "move_speed_j(jnt[";
  for (std::size_t index = 0; index < joint_velocity_deg_s.size(); ++index) {
    if (index > 0U) {
      command_stream << ",";
    }
    command_stream << joint_velocity_deg_s[index];
  }
  command_stream << "],";
  command_stream << std::setprecision(6)
                 << time1 << "," << time2 << "," << gain << "," << lpf_gain << ")";

  return send_command(command_stream.str());
}
```

의미:

- rad/s가 아니라 deg/s 배열을 받아 RB10 API 문자열인 `move_speed_j(jnt[...],t1,t2,gain,alpha)` 형태로 만든다.
- controller 쪽에서 rad/s를 deg/s로 변환한 뒤 이 함수를 호출한다.

### 6. Python API bridge에서 velocity 명령 처리

파일: `scripts/rb10_api_bridge.py`

```python
def _normalize_command_mode(value):
    normalized = str(value).strip().lower()
    if normalized in {'position', 'servo_j', 'servoj'}:
        return 'position'
    if normalized in {'velocity', 'speed_j', 'speedj'}:
        return 'velocity'
    raise ValueError('command_mode must be one of: position, velocity, servo_j, speed_j')
```

```python
if self.command_mode == 'velocity':
    desired_velocity_deg_s = [math.degrees(float(j)) for j in msg.data[:6]]
    safe_velocity_deg_s = []
    clamped = False
    for desired_deg_s in desired_velocity_deg_s:
        limited_deg_s = max(
            -self.max_command_velocity_deg_s,
            min(self.max_command_velocity_deg_s, desired_deg_s),
        )
        safe_velocity_deg_s.append(limited_deg_s)
        if limited_deg_s != desired_deg_s:
            clamped = True

    cmd = (
        'move_speed_j(jnt[' + ','.join(f'{j:.3f}' for j in safe_velocity_deg_s) + '],' +
        f'{self.speedj_t1},{self.speedj_t2},{self.speedj_gain},{self.speedj_alpha})'
    )
    self.api['SendCOMMAND'](cmd, self.api['CMD_TYPE'].MOVE)
    return
```

의미:

- `/position_controllers/commands`로 들어온 6개 값을 velocity 모드에서는 rad/s로 해석한다.
- Python bridge 안에서 deg/s로 바꾼 뒤 `move_speed_j`를 보낸다.
- 속도 제한도 한 번 더 적용한다.

### 7. cobot.py에서 move_speed_j 허용

파일: `scripts/api/cobot.py`

```python
if 'move_servo_j' in str or 'move_speed_j' in str:
    CMDSock.send(str_space.encode('utf-8'))
    return True
```

의미:

- 기존 즉시 송신 예외 처리에 `move_speed_j`를 추가했다.
- 이 변경이 없으면 Python API bridge에서 만든 speedj 명령이 기존 command wrapper 흐름에 걸릴 수 있다.

### 8. launch 파일의 command_mode 인자

파일: `launch/rb10_rmpflow.launch.py`

```python
DeclareLaunchArgument(
    "command_mode",
    default_value="position",
    description="RB10 command output mode: position sends move_servo_j, velocity sends move_speed_j.",
)
```

```python
"command_mode": command_mode,
```

의미:

- 런치에서 `command_mode:=velocity`처럼 모드를 선택할 수 있다.
- controller와 API bridge 쪽 파라미터로 전달된다.

예시:

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py command_mode:=velocity
```

### 9. params.yaml 추가/변경

파일: `config/params.yaml`

```yaml
command_mode: "position"
speedj_t1: 0.01
speedj_t2: 0.1
speedj_gain: 0.5
speedj_alpha: 0.5
```

```yaml
rb10_api_bridge:
  ros__parameters:
    speedj_t1: 0.002
    speedj_t2: 0.1
    speedj_gain: 0.005
    speedj_alpha: 0.1

rb10_direct_bridge:
  ros__parameters:
    command_mode: "position"
    speedj_t1: 0.01
    speedj_t2: 0.1
    speedj_gain: 0.5
    speedj_alpha: 0.5
```

튜닝 변경:

```yaml
collision_rmp_repulsion_gain: 300.0
target_rmp_accel_p_gain: 30.0
target_rmp_accel_d_gain: 100.0
damping_rmp_accel_d_gain: 80.0
damping_rmp_inertia: 300.0
command_guard_max_velocity_rad_s: 2.0
control_rate: 200.0
synced_input_velocity_filter_alpha: 0.7
synced_input_velocity_filter_beta: 0.04
measured_position_feedback_blend: 1.0
measured_velocity_feedback_blend: 0.6
```

의미:

- RMP 출력이 너무 공격적이지 않도록 repulsion/target/damping 값을 조정했다.
- 제어 주기를 200 Hz로 올렸다.
- velocity 모드에서 사용할 `speedj_*` 계열 파라미터를 추가했다.

## 실행 조합별 의미

### 1. RViz/토픽 출력만 확인

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py \
  use_direct_hardware_backend:=false \
  start_api_bridge:=false
```

이 경우 controller는 joint command topic 쪽으로 결과를 publish하지만, RB10 API로 직접 보내지는 않는다.

### 2. direct backend로 RB10 API 직접 송신

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py \
  use_direct_hardware_backend:=true \
  command_mode:=position
```

position 모드에서는 `move_servo_j`가 나간다.

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py \
  use_direct_hardware_backend:=true \
  command_mode:=velocity
```

velocity 모드에서는 `move_speed_j`가 나간다.

### 3. Python API bridge를 통한 송신

```bash
ros2 launch rb10_rmpflow_rviz rb10_rmpflow.launch.py \
  use_direct_hardware_backend:=false \
  start_api_bridge:=true \
  command_mode:=velocity
```

이 경우 controller가 topic으로 `qd`를 publish하고, `rb10_api_bridge.py`가 그 값을 받아 `move_speed_j`로 변환해서 API 송신한다.

## 확인 포인트

다음 명령으로 현재 controller가 어떤 모드로 떠 있는지 확인할 수 있다.

```bash
ros2 param get /rmpflow_controller command_mode
ros2 param get /rmpflow_controller backend_mode
```

명령이 실제로 어떤 topic으로 나가는지 확인하려면:

```bash
ros2 topic info /position_controllers/commands -v
ros2 topic echo /position_controllers/commands --once
```

RB10 direct backend를 쓰는 경우에는 topic publish가 아니라 controller 내부에서 socket client를 통해 직접 API 명령을 보낸다.

## 빌드 상태

이 문서는 현재 워킹트리 변경 사항을 설명하기 위해 추가한 문서 파일이다. 문서 작성 자체로는 빌드를 다시 수행하지 않았다.


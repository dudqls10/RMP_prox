#include <pthread.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_box.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "rb10_rmpflow_rviz/rb10_socket_client.hpp"
#include "rb10_rmpflow_rviz/pinocchio_direct_solver.hpp"
#include "rb10_rmpflow_rviz/rb10_model.hpp"
#include "rb10_rmpflow_rviz/rmp_solver_interface.hpp"

namespace rb10_rmpflow_rviz
{

namespace
{

using JointVector = RB10Model::JointVector;
using JointStateRtPublisher = realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>;
using Float64ArrayRtPublisher =
  realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>;
using PoseRtPublisher = realtime_tools::RealtimePublisher<geometry_msgs::msg::Pose>;
constexpr double kPi = 3.14159265358979323846;

double degrees_to_radians(double degrees)
{
  return degrees * kPi / 180.0;
}

double radians_to_degrees(double radians)
{
  return radians * 180.0 / kPi;
}

struct RobotState
{
  JointVector q{JointVector::Zero()};
  JointVector qd{JointVector::Zero()};
};

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

struct TcpAccelerationSample
{
  std::array<double, 3> acceleration{0.0, 0.0, 0.0};
  double norm{0.0};
  bool valid{false};
};

struct TangentEscapeDebugSample
{
  std::size_t control_point_index{0};
  Eigen::Vector3d control_point{Eigen::Vector3d::Zero()};
  Eigen::Vector3d obstacle_center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d tangent{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d raw_accel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d filtered_accel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d raw_tcp_accel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d filtered_tcp_accel{Eigen::Vector3d::Zero()};
  JointVector qdd_raw{JointVector::Zero()};
  JointVector qdd_filtered{JointVector::Zero()};
  double clearance{0.0};
  double activation{0.0};
  double score{0.0};
  bool has_tangent{false};
};

struct TangentEscapeDebugState
{
  std::vector<TangentEscapeDebugSample> samples;
};

class SynchronizedVelocityFilter
{
public:
  enum class FilterType
  {
    kMovingAverage,
    kAlphaBeta
  };

  static FilterType parse_filter_type(const std::string & filter_type)
  {
    if (
      filter_type == "alphabeta" ||
      filter_type == "alpha-beta" ||
      filter_type == "alpha_beta")
    {
      return FilterType::kAlphaBeta;
    }
    return FilterType::kMovingAverage;
  }

  static const char * filter_type_to_string(FilterType filter_type)
  {
    if (filter_type == FilterType::kAlphaBeta) {
      return "alpha-beta";
    }
    return "moving-average";
  }

  SynchronizedVelocityFilter(
    double input_rate_hz,
    double output_rate_hz,
    double lowpass_alpha,
    double ratio_tolerance,
    FilterType filter_type = FilterType::kMovingAverage,
    double alpha_beta_beta = 0.0)
  : input_rate_hz_(std::max(input_rate_hz, 1.0)),
    output_rate_hz_(std::max(output_rate_hz, 1.0)),
    lowpass_alpha_(std::clamp(lowpass_alpha, 0.0, 1.0)),
    ratio_tolerance_(std::max(ratio_tolerance, 0.0)),
    filter_type_(filter_type),
    alpha_beta_beta_(std::clamp(alpha_beta_beta, 0.0, 1.0))
  {
    const double raw_ratio = input_rate_hz_ / output_rate_hz_;
    const double rounded_ratio = std::round(raw_ratio);
    if (
      rounded_ratio >= 1.0 &&
      std::abs(raw_ratio - rounded_ratio) <= ratio_tolerance_)
    {
      sync_multiple_ = static_cast<int>(rounded_ratio);
      integer_ratio_aligned_ = true;
    } else {
      sync_multiple_ = std::max(1, static_cast<int>(std::lround(std::max(raw_ratio, 1.0))));
      integer_ratio_aligned_ = false;
    }
  }

  void reset(
    const JointVector & position,
    const std::chrono::steady_clock::time_point & current_time)
  {
    previous_position_ = position;
    previous_time_ = current_time;
    accumulated_velocity_.setZero();
    filtered_velocity_.setZero();
    accumulated_time_sec_ = 0.0;
    alpha_beta_position_.setZero();
    alpha_beta_initialized_ = false;
    accumulated_samples_ = 0;
    initialized_ = true;
    output_initialized_ = false;
  }

  JointVector update(
    const JointVector & position,
    const std::chrono::steady_clock::time_point & current_time)
  {
    if (!initialized_) {
      reset(position, current_time);
      return JointVector::Zero();
    }

    const std::chrono::duration<double> dt_duration = current_time - previous_time_;
    const double dt_sec = dt_duration.count();
    if (dt_sec <= std::numeric_limits<double>::epsilon() || dt_sec > 0.1) {
      reset(position, current_time);
      return JointVector::Zero();
    }

    const JointVector raw_velocity = (position - previous_position_) / dt_sec;
    previous_position_ = position;
    previous_time_ = current_time;
    accumulated_velocity_ += raw_velocity;
    accumulated_time_sec_ += dt_sec;
    ++accumulated_samples_;

    if (accumulated_samples_ >= sync_multiple_) {
      const JointVector measured_velocity =
        accumulated_velocity_ / static_cast<double>(accumulated_samples_);
      const double synced_dt_sec =
        std::max(accumulated_time_sec_, std::numeric_limits<double>::epsilon());

      if (filter_type_ == FilterType::kAlphaBeta) {
        if (!alpha_beta_initialized_) {
          alpha_beta_position_ = position;
          filtered_velocity_ = measured_velocity;
          alpha_beta_initialized_ = true;
        } else {
          const JointVector predicted_position =
            alpha_beta_position_ + filtered_velocity_ * synced_dt_sec;
          const JointVector residual = position - predicted_position;
          alpha_beta_position_ = predicted_position + lowpass_alpha_ * residual;
          filtered_velocity_ += (alpha_beta_beta_ * residual) / synced_dt_sec;
        }
      } else {
        if (!output_initialized_) {
          filtered_velocity_ = measured_velocity;
        } else {
          filtered_velocity_ =
            lowpass_alpha_ * measured_velocity +
            (1.0 - lowpass_alpha_) * filtered_velocity_;
        }
      }
      output_initialized_ = true;
      accumulated_velocity_.setZero();
      accumulated_time_sec_ = 0.0;
      accumulated_samples_ = 0;
    }

    if (!output_initialized_) {
      return JointVector::Zero();
    }
    return filtered_velocity_;
  }

  int sync_multiple() const
  {
    return sync_multiple_;
  }

  bool integer_ratio_aligned() const
  {
    return integer_ratio_aligned_;
  }

private:
  double input_rate_hz_{100.0};
  double output_rate_hz_{100.0};
  double lowpass_alpha_{0.25};
  double ratio_tolerance_{0.05};
  FilterType filter_type_{FilterType::kMovingAverage};
  double alpha_beta_beta_{0.0};
  bool alpha_beta_initialized_{false};
  JointVector alpha_beta_position_{JointVector::Zero()};
  int sync_multiple_{1};
  bool integer_ratio_aligned_{true};
  bool initialized_{false};
  bool output_initialized_{false};
  int accumulated_samples_{0};
  JointVector previous_position_{JointVector::Zero()};
  JointVector accumulated_velocity_{JointVector::Zero()};
  JointVector filtered_velocity_{JointVector::Zero()};
  std::chrono::steady_clock::time_point previous_time_{};
  double accumulated_time_sec_{0.0};
};

struct ExternalRmpBuffer
{
  int dim{0};
  Eigen::MatrixXd metric_sqrt{Eigen::MatrixXd::Zero(0, 0)};
  Eigen::VectorXd acceleration{Eigen::VectorXd::Zero(0)};
  bool has_metric{false};
  bool has_acceleration{false};
};

struct GoalTarget
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

bool command_blocked_by_robot_safety(const Rb10SystemState & state)
{
  return state.op_stat_soft_estop_occur != 0 ||
         state.op_stat_ems_flag != 0 ||
         state.op_stat_collision_occur != 0 ||
         state.op_stat_self_collision != 0 ||
         state.op_stat_sos_flag != 0;
}

std::string robot_safety_summary(const Rb10SystemState & state)
{
  std::string summary;
  const auto append_flag = [&summary](const char * label, std::int32_t value) {
    if (value == 0) {
      return;
    }
    if (!summary.empty()) {
      summary += ", ";
    }
    summary += label;
  };

  append_flag("soft_estop", state.op_stat_soft_estop_occur);
  append_flag("ems", state.op_stat_ems_flag);
  append_flag("collision", state.op_stat_collision_occur);
  append_flag("self_collision", state.op_stat_self_collision);
  append_flag("sos", state.op_stat_sos_flag);

  if (summary.empty()) {
    summary = "none";
  }
  return summary;
}

class ControllerBackend
{
public:
  virtual ~ControllerBackend() = default;

  virtual RobotState read_state() = 0;
  virtual bool ready() const
  {
    return true;
  }
  virtual void apply_command(
    const RobotState & command_state,
    const std::array<const char *, 6> & joint_names) = 0;
};

bool joint_state_to_robot_state(const sensor_msgs::msg::JointState & msg, RobotState & state)
{
  if (msg.position.size() < RB10Model::joint_names.size()) {
    return false;
  }

  if (msg.name.size() >= RB10Model::joint_names.size()) {
    std::unordered_map<std::string, std::size_t> name_to_index;
    name_to_index.reserve(msg.name.size());
    for (std::size_t index = 0; index < msg.name.size(); ++index) {
      name_to_index.emplace(msg.name[index], index);
    }

    bool matched_all = true;
    for (std::size_t index = 0; index < RB10Model::joint_names.size(); ++index) {
      const auto it = name_to_index.find(RB10Model::joint_names[index]);
      if (it == name_to_index.end() || it->second >= msg.position.size()) {
        matched_all = false;
        break;
      }
      state.q[static_cast<int>(index)] = msg.position[it->second];
      state.qd[static_cast<int>(index)] =
        it->second < msg.velocity.size() ? msg.velocity[it->second] : 0.0;
    }

    if (matched_all) {
      return true;
    }
  }

  for (std::size_t index = 0; index < RB10Model::joint_names.size(); ++index) {
    state.q[static_cast<int>(index)] = msg.position[index];
    state.qd[static_cast<int>(index)] =
      index < msg.velocity.size() ? msg.velocity[index] : 0.0;
  }
  return true;
}

class SimulationBackend : public ControllerBackend
{
public:
  explicit SimulationBackend(const JointVector & initial_q)
  {
    state_.q = initial_q;
  }

  RobotState read_state() override
  {
    std::scoped_lock lock(mutex_);
    return state_;
  }

  void apply_command(
    const RobotState & command_state,
    const std::array<const char *, 6> &) override
  {
    std::scoped_lock lock(mutex_);
    state_ = command_state;
  }

private:
  std::mutex mutex_;
  RobotState state_;
};

class HardwareBridgeBackend : public ControllerBackend
{
public:
  HardwareBridgeBackend(
    rclcpp::Node * node,
    const JointVector & initial_q,
    const std::string & state_topic,
    const std::string & command_topic)
  : node_(node)
  {
    RobotState initial_state;
    initial_state.q = initial_q;
    state_buffer_.initRT(initial_state);
    command_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(command_topic, 10);
    state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
      state_topic,
      10,
      std::bind(&HardwareBridgeBackend::on_state, this, std::placeholders::_1));
  }

  RobotState read_state() override
  {
    return *state_buffer_.readFromRT();
  }

  void apply_command(
    const RobotState & command_state,
    const std::array<const char *, 6> & joint_names) override
  {
    sensor_msgs::msg::JointState command;
    command.header.stamp = node_->now();
    command.name.assign(joint_names.begin(), joint_names.end());
    command.position.assign(command_state.q.data(), command_state.q.data() + command_state.q.size());
    command.velocity.assign(command_state.qd.data(), command_state.qd.data() + command_state.qd.size());

    command_pub_->publish(command);
  }

private:
  void on_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    RobotState next_state;
    if (!joint_state_to_robot_state(*msg, next_state)) {
      return;
    }

    state_buffer_.writeFromNonRT(next_state);
  }

  rclcpp::Node * node_;
  realtime_tools::RealtimeBuffer<RobotState> state_buffer_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr command_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
};

class JointCommandTopicsBackend : public ControllerBackend
{
public:
  JointCommandTopicsBackend(
    rclcpp::Node * node,
    const JointVector & initial_q,
    const std::string & state_topic,
    const std::string & command_topic,
    CommandMode command_mode)
  : node_(node)
  , state_topic_(state_topic)
  , command_mode_(command_mode)
  {
    RobotState initial_state;
    initial_state.q = initial_q;
    state_buffer_.initRT(initial_state);
    command_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(command_topic, 10);
    state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
      state_topic,
      10,
      std::bind(&JointCommandTopicsBackend::on_state, this, std::placeholders::_1));
  }

  RobotState read_state() override
  {
    return *state_buffer_.readFromRT();
  }

  bool ready() const override
  {
    return state_received_.load();
  }

  void apply_command(
    const RobotState & command_state,
    const std::array<const char *, 6> &) override
  {
    if (!ready()) {
      return;
    }

    std_msgs::msg::Float64MultiArray command;
    const JointVector & command_vector =
      command_mode_ == CommandMode::kVelocity ? command_state.qd : command_state.q;
    command.data.assign(command_vector.data(), command_vector.data() + command_vector.size());

    command_pub_->publish(command);
  }

private:
  void on_state(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    RobotState next_state;
    if (!joint_state_to_robot_state(*msg, next_state)) {
      return;
    }

    state_buffer_.writeFromNonRT(next_state);
    state_received_.store(true);
    if (!logged_initial_state_) {
      RCLCPP_INFO(
        node_->get_logger(),
        "Received initial robot state on %s; enabling RMP commands",
        state_topic_.c_str());
      logged_initial_state_ = true;
    }
  }

  rclcpp::Node * node_;
  std::string state_topic_;
  CommandMode command_mode_{CommandMode::kPosition};
  realtime_tools::RealtimeBuffer<RobotState> state_buffer_;
  std::atomic<bool> state_received_{false};
  bool logged_initial_state_{false};
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr command_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
};

class DirectRb10Backend : public ControllerBackend
{
public:
  DirectRb10Backend(
    rclcpp::Node * node,
    const JointVector & initial_q,
    const std::string & robot_ip,
    bool simulation_mode,
    const std::string & real_joint_state_source,
    bool publish_debug_joint_state_sources,
    CommandMode command_mode,
    double data_request_rate_hz,
    double control_rate_hz,
    double servo_t1,
    double servo_t2,
    double servo_gain,
    double servo_alpha,
    double speedj_t1,
    double speedj_t2,
    double speedj_gain,
    double speedj_alpha,
    bool use_synced_input_velocity_filter,
    double synced_input_velocity_filter_alpha,
    double synced_input_velocity_filter_beta,
    const std::string & synced_input_velocity_filter_type,
    double synced_input_velocity_ratio_tolerance,
    bool startup_move_to_default_pose,
    const std::vector<double> & startup_home_joints_deg,
    double startup_movej_speed,
    double startup_movej_accel,
    double startup_release_tolerance_deg,
    double startup_release_timeout_sec,
    bool enable_socket_realtime,
    int socket_realtime_priority,
    bool stop_on_shutdown,
    const std::string & shutdown_action)
  : node_(node),
    simulation_mode_(simulation_mode),
    real_joint_state_source_(real_joint_state_source),
    publish_debug_joint_state_sources_(publish_debug_joint_state_sources),
    command_mode_(command_mode),
    servo_t1_(servo_t1),
    servo_t2_(servo_t2),
    servo_gain_(servo_gain),
    servo_alpha_(servo_alpha),
    speedj_t1_(speedj_t1),
    speedj_t2_(speedj_t2),
    speedj_gain_(speedj_gain),
    speedj_alpha_(speedj_alpha),
    use_synced_input_velocity_filter_(use_synced_input_velocity_filter),
    startup_move_to_default_pose_(startup_move_to_default_pose),
    startup_movej_speed_(startup_movej_speed),
    startup_movej_accel_(startup_movej_accel),
    startup_release_tolerance_deg_(startup_release_tolerance_deg),
    startup_release_timeout_sec_(startup_release_timeout_sec),
    stop_on_shutdown_(stop_on_shutdown),
    shutdown_action_(shutdown_action)
  {
    RobotState initial_state;
    initial_state.q = initial_q;
    state_buffer_.initRT(initial_state);
    if (startup_home_joints_deg.size() != 6U) {
      throw std::runtime_error("startup_home_joints_deg must contain exactly 6 values");
    }
    for (std::size_t index = 0; index < 6U; ++index) {
      startup_home_joints_deg_[index] = startup_home_joints_deg[index];
    }

    if (!socket_client_.connect(
        robot_ip,
        data_request_rate_hz,
        std::bind(&DirectRb10Backend::on_state, this, std::placeholders::_1),
        std::bind(&DirectRb10Backend::on_cmd_log, this, std::placeholders::_1),
        enable_socket_realtime,
        socket_realtime_priority))
    {
      throw std::runtime_error("Failed to open RB10 command/data sockets from direct backend");
    }

    if (!socket_client_.initialize_robot(simulation_mode_)) {
      socket_client_.disconnect();
      throw std::runtime_error("Failed to initialize RB10 from direct backend");
    }

    if (use_synced_input_velocity_filter_) {
      const auto filter_type =
        SynchronizedVelocityFilter::parse_filter_type(synced_input_velocity_filter_type);
      synced_input_velocity_filter_ = std::make_unique<SynchronizedVelocityFilter>(
        data_request_rate_hz,
        control_rate_hz,
        synced_input_velocity_filter_alpha,
        synced_input_velocity_ratio_tolerance,
        filter_type,
        synced_input_velocity_filter_beta);
      RCLCPP_INFO(
        node_->get_logger(),
        "Direct RB10 backend high-rate velocity sync filter enabled (%s, alpha=%.3f beta=%.3f): %.1f Hz input -> %d-sample synced velocity for %.1f Hz control",
        SynchronizedVelocityFilter::filter_type_to_string(filter_type),
        synced_input_velocity_filter_alpha,
        synced_input_velocity_filter_beta,
        data_request_rate_hz,
        synced_input_velocity_filter_->sync_multiple(),
        control_rate_hz);
      if (!synced_input_velocity_filter_->integer_ratio_aligned()) {
        RCLCPP_WARN(
          node_->get_logger(),
          "hardware_data_request_rate/control_rate is not close to an integer multiple; using the nearest %d-sample velocity sync window",
          synced_input_velocity_filter_->sync_multiple());
      }
    }

    if (publish_debug_joint_state_sources_) {
      reference_joint_state_publisher_ =
        node_->create_publisher<sensor_msgs::msg::JointState>("/rb10/reference_joint_states", 10);
      measured_joint_state_publisher_ =
        node_->create_publisher<sensor_msgs::msg::JointState>("/rb10/measured_joint_states", 10);
      tracking_error_publisher_ =
        node_->create_publisher<std_msgs::msg::Float64MultiArray>("/rb10/joint_tracking_error_deg", 10);
    }

    if (startup_move_to_default_pose_) {
      if (!socket_client_.send_movej_degrees(
          startup_home_joints_deg_, startup_movej_speed_, startup_movej_accel_))
      {
        socket_client_.disconnect();
        throw std::runtime_error("Failed to command RB10 startup move_j from direct backend");
      }
      startup_waiting_for_home_.store(true);
      startup_release_deadline_ =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(startup_release_timeout_sec_));
      RCLCPP_INFO(
        node_->get_logger(),
        "Direct RB10 backend commanded startup move_j; waiting to release the control loop until home is reached or timeout expires");
    }

    servo_command_thread_ = std::thread(&DirectRb10Backend::servo_command_loop, this);
  }

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
    if (stop_on_shutdown_ && socket_client_.is_connected()) {
      const bool halt_first = shutdown_action_ != "pause";
      const bool stopped = socket_client_.send_shutdown_sequence(halt_first);
      if (stopped) {
        RCLCPP_INFO(node_->get_logger(), "Direct RB10 backend sent shutdown stop/clear sequence");
      } else {
        RCLCPP_WARN(node_->get_logger(), "Direct RB10 backend shutdown sequence did not complete cleanly");
      }
    }
    socket_client_.disconnect();
  }

  RobotState read_state() override
  {
    return *state_buffer_.readFromRT();
  }

  bool ready() const override
  {
    return socket_client_.is_connected() &&
           state_received_.load() &&
           !startup_waiting_for_home_.load();
  }

  void apply_command(
    const RobotState & command_state,
    const std::array<const char *, 6> &) override
  {
    if (!ready()) {
      return;
    }
    if (command_blocked_due_to_safety_.load()) {
      RCLCPP_WARN_THROTTLE(
        node_->get_logger(),
        *node_->get_clock(),
        1000,
        "Direct RB10 backend is blocking %s while a robot safety stop is active",
        rb10_stream_command_name(command_mode_));
      return;
    }

    std::array<double, 6> joint_deg{};
    const JointVector & command_vector =
      command_mode_ == CommandMode::kVelocity ? command_state.qd : command_state.q;
    for (std::size_t index = 0; index < 6U; ++index) {
      joint_deg[index] = radians_to_degrees(command_vector[static_cast<int>(index)]);
    }
    queue_servo_command(joint_deg);
  }

private:
  void queue_servo_command(const std::array<double, 6> & joint_deg)
  {
    bool notify = false;
    {
      std::scoped_lock lock(servo_command_wait_mutex_);
      latest_servo_joint_deg_.set(joint_deg);
      servo_command_pending_ = true;
      notify = true;
    }
    if (notify) {
      servo_command_cv_.notify_one();
    }
  }

  void clear_pending_servo_command()
  {
    std::scoped_lock lock(servo_command_wait_mutex_);
    servo_command_pending_ = false;
  }

  void stop_servo_command_thread()
  {
    {
      std::scoped_lock lock(servo_command_wait_mutex_);
      servo_command_thread_running_ = false;
      servo_command_pending_ = false;
    }
    servo_command_cv_.notify_all();
    if (servo_command_thread_.joinable()) {
      servo_command_thread_.join();
    }
  }

  void servo_command_loop()
  {
    while (true) {
      std::array<double, 6> joint_deg{};
      {
        std::unique_lock<std::mutex> lock(servo_command_wait_mutex_);
        servo_command_cv_.wait(lock, [this]() {
          return !servo_command_thread_running_ || servo_command_pending_;
        });
        if (!servo_command_thread_running_ && !servo_command_pending_) {
          return;
        }
        servo_command_pending_ = false;
        latest_servo_joint_deg_.get(joint_deg);
      }

      const bool sent =
        command_mode_ == CommandMode::kVelocity ?
        socket_client_.send_speedj_degrees_per_sec(
          joint_deg, speedj_t1_, speedj_t2_, speedj_gain_, speedj_alpha_) :
        socket_client_.send_servoj_degrees(
          joint_deg, servo_t1_, servo_t2_, servo_gain_, servo_alpha_);
      if (!sent)
      {
        RCLCPP_WARN_THROTTLE(
          node_->get_logger(),
          *node_->get_clock(),
          1000,
          "Direct RB10 backend failed to send %s command",
          rb10_stream_command_name(command_mode_));
      }
    }
  }

  void on_state(const Rb10SystemState & state)
  {
    RobotState next_state;
    const auto current_time = std::chrono::steady_clock::now();
    std::array<double, 6> reference_position_rad{};
    std::array<double, 6> measured_position_rad{};
    const auto & joint_source_deg = select_joint_source_deg(state);
    for (std::size_t index = 0; index < 6U; ++index) {
      reference_position_rad[index] = degrees_to_radians(state.joint_ref_deg[index]);
      measured_position_rad[index] = degrees_to_radians(state.joint_ang_deg[index]);
      next_state.q[static_cast<int>(index)] = degrees_to_radians(joint_source_deg[index]);
    }

    if (use_synced_input_velocity_filter_ && synced_input_velocity_filter_) {
      next_state.qd = synced_input_velocity_filter_->update(next_state.q, current_time);
    }

    state_buffer_.writeFromNonRT(next_state);
    state_received_.store(true);

    const bool command_blocked = command_blocked_by_robot_safety(state);
    const bool was_command_blocked = command_blocked_due_to_safety_.exchange(command_blocked);
    if (command_blocked && !was_command_blocked) {
      clear_pending_servo_command();
      RCLCPP_ERROR(
        node_->get_logger(),
        "RB10 safety stop active; blocking %s commands (%s)",
        rb10_stream_command_name(command_mode_),
        robot_safety_summary(state).c_str());
    } else if (!command_blocked && was_command_blocked) {
      RCLCPP_INFO(node_->get_logger(), "RB10 safety stop cleared; servo commands may resume");
    }

    if (publish_debug_joint_state_sources_) {
      publish_debug_joint_states(reference_position_rad, measured_position_rad);
    }

    if (!startup_waiting_for_home_.load()) {
      return;
    }

    bool home_reached = true;
    const auto & startup_joint_deg =
      simulation_mode_ ? state.joint_ref_deg : state.joint_ang_deg;
    for (std::size_t index = 0; index < 6U; ++index) {
      if (
        std::fabs(startup_joint_deg[index] - startup_home_joints_deg_[index]) >
        startup_release_tolerance_deg_)
      {
        home_reached = false;
        break;
      }
    }

    if (home_reached || current_time >= startup_release_deadline_) {
      startup_waiting_for_home_.store(false);
      if (use_synced_input_velocity_filter_ && synced_input_velocity_filter_) {
        synced_input_velocity_filter_->reset(next_state.q, current_time);
        next_state.qd.setZero();
        state_buffer_.writeFromNonRT(next_state);
      }
      if (home_reached) {
        RCLCPP_INFO(node_->get_logger(), "Direct RB10 backend startup home pose reached; enabling RMP commands");
      } else {
        RCLCPP_WARN(node_->get_logger(), "Direct RB10 backend startup release timeout expired; enabling RMP commands");
      }
    }
  }

  void on_cmd_log(const std::string & text)
  {
    if (text.empty()) {
      return;
    }
    if (text.rfind("[socket]", 0U) == 0U) {
      RCLCPP_WARN(node_->get_logger(), "%s", text.c_str());
      return;
    }
    if (text == "The command was executed") {
      RCLCPP_DEBUG(node_->get_logger(), "RB10 CMD: %s", text.c_str());
      return;
    }
    RCLCPP_INFO(node_->get_logger(), "RB10 CMD: %s", text.c_str());
  }

  void publish_debug_joint_states(
    const std::array<double, 6> & reference_position_rad,
    const std::array<double, 6> & measured_position_rad)
  {
    const auto stamp = node_->now();

    if (reference_joint_state_publisher_) {
      sensor_msgs::msg::JointState ref_msg;
      ref_msg.header.stamp = stamp;
      ref_msg.name.assign(RB10Model::joint_names.begin(), RB10Model::joint_names.end());
      ref_msg.position.assign(reference_position_rad.begin(), reference_position_rad.end());
      reference_joint_state_publisher_->publish(ref_msg);
    }

    if (measured_joint_state_publisher_) {
      sensor_msgs::msg::JointState measured_msg;
      measured_msg.header.stamp = stamp;
      measured_msg.name.assign(RB10Model::joint_names.begin(), RB10Model::joint_names.end());
      measured_msg.position.assign(measured_position_rad.begin(), measured_position_rad.end());
      measured_joint_state_publisher_->publish(measured_msg);
    }

    if (tracking_error_publisher_) {
      std_msgs::msg::Float64MultiArray tracking_error;
      tracking_error.data.resize(6U);
      for (std::size_t index = 0; index < 6U; ++index) {
        tracking_error.data[index] =
          radians_to_degrees(reference_position_rad[index] - measured_position_rad[index]);
      }
      tracking_error_publisher_->publish(tracking_error);
    }
  }

  const std::array<float, 6> & select_joint_source_deg(const Rb10SystemState & state) const
  {
    if (simulation_mode_) {
      return state.joint_ref_deg;
    }
    if (real_joint_state_source_ == "reference") {
      return state.joint_ref_deg;
    }
    return state.joint_ang_deg;
  }

  rclcpp::Node * node_;
  realtime_tools::RealtimeBuffer<RobotState> state_buffer_;
  Rb10SocketClient socket_client_;
  bool simulation_mode_{false};
  std::string real_joint_state_source_{"measured"};
  bool publish_debug_joint_state_sources_{false};
  CommandMode command_mode_{CommandMode::kPosition};
  double servo_t1_{0.002};
  double servo_t2_{0.1};
  double servo_gain_{0.02};
  double servo_alpha_{0.2};
  double speedj_t1_{0.02};
  double speedj_t2_{0.2};
  double speedj_gain_{0.05};
  double speedj_alpha_{0.1};
  bool use_synced_input_velocity_filter_{false};
  bool startup_move_to_default_pose_{true};
  std::array<double, 6> startup_home_joints_deg_{};
  double startup_movej_speed_{20.0};
  double startup_movej_accel_{20.0};
  double startup_release_tolerance_deg_{2.0};
  double startup_release_timeout_sec_{12.0};
  std::atomic<bool> startup_waiting_for_home_{false};
  std::chrono::steady_clock::time_point startup_release_deadline_{};
  bool stop_on_shutdown_{true};
  std::string shutdown_action_{"halt"};
  std::atomic<bool> state_received_{false};
  std::atomic<bool> command_blocked_due_to_safety_{false};
  realtime_tools::RealtimeBox<std::array<double, 6>> latest_servo_joint_deg_;
  std::mutex servo_command_wait_mutex_;
  std::condition_variable servo_command_cv_;
  std::thread servo_command_thread_;
  bool servo_command_pending_{false};
  bool servo_command_thread_running_{true};
  std::unique_ptr<SynchronizedVelocityFilter> synced_input_velocity_filter_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr reference_joint_state_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr measured_joint_state_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr tracking_error_publisher_;
};

bool enable_realtime(
  rclcpp::Logger logger,
  bool enabled,
  int priority,
  bool lock_memory)
{
  if (!enabled) {
    return false;
  }

  if (lock_memory && mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    RCLCPP_WARN(logger, "mlockall failed; continuing without memory locking");
  }

  sched_param params{};
  params.sched_priority = priority;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &params) != 0) {
    RCLCPP_WARN(logger, "Failed to switch control thread to SCHED_FIFO priority %d", priority);
    return false;
  }

  RCLCPP_INFO(logger, "Control thread running with SCHED_FIFO priority %d", priority);
  return true;
}

class RmpflowControllerNode : public rclcpp::Node
{
public:
  RmpflowControllerNode()
  : Node("rmpflow_controller"), running_(true)
  {
    declare_parameter("control_rate", 100.0);
    declare_parameter("visualization_rate", 20.0);
    declare_parameter("goal_x", 0.6);
    declare_parameter("goal_y", -0.4);
    declare_parameter("goal_z", 0.6);
    declare_parameter("min_goal_z", 0.05);
    declare_parameter("initialize_goal_from_first_state", true);
    declare_parameter("safety_stop_on_min_z", true);
    declare_parameter("workspace_floor_z", 0.0);
    declare_parameter("min_link6_z", 0.03);
    declare_parameter("min_tcp_z", 0.05);
    declare_parameter("body_goal", std::vector<double>{0.45, 0.0, 0.9});
    declare_parameter("orientation_goal", std::vector<double>{0.0, 0.0, 1.0});
    declare_parameter("initial_q", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    declare_parameter("default_q", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    declare_parameter(
      "joint_lower_limits_deg",
      std::vector<double>{-180.0, -180.0, -180.0, -180.0, -180.0, -180.0});
    declare_parameter(
      "joint_upper_limits_deg",
      std::vector<double>{180.0, 180.0, 180.0, 180.0, 180.0, 180.0});
    declare_parameter("joint_limit_buffers", std::vector<double>{0.01, 0.01, 0.01, 0.01, 0.01, 0.01});
    declare_parameter("backend_mode", "simulation");
    declare_parameter("command_mode", "velocity");
    declare_parameter("robot_ip", "192.168.111.50");
    declare_parameter("simulation_mode", false);
    declare_parameter("real_joint_state_source", "measured");
    declare_parameter("publish_debug_joint_state_sources", false);
    declare_parameter("hardware_data_request_rate", 500.0);
    declare_parameter("use_synced_input_velocity_filter", false);
    declare_parameter("synced_input_velocity_filter_alpha", 0.35);
    declare_parameter("synced_input_velocity_filter_beta", 0.02);
    declare_parameter("synced_input_velocity_filter_type", std::string("alpha-beta"));
    declare_parameter("synced_input_velocity_ratio_tolerance", 0.05);
    declare_parameter("hardware_state_topic", "hardware_joint_states");
    declare_parameter("hardware_command_topic", "hardware_joint_command");
    declare_parameter("joint_state_topic", "joint_states");
    declare_parameter("rmp_flag_gate_enabled", false);
    declare_parameter("rmp_flag_topic", "/RMP_flag");
    declare_parameter("rmp_active_flag_value", 1);
    declare_parameter("position_command_topic", "position_controllers/commands");
    declare_parameter("publish_position_command", true);
    declare_parameter("position_command_state_topic", "/rmp_position_command");
    declare_parameter("publish_target_q", false);
    declare_parameter("target_q_topic", "/target_q");
    declare_parameter("publish_target_metric", false);
    declare_parameter("target_metric_topic", "/target_metric");
    declare_parameter("publish_rmp_accel_debug", true);
    declare_parameter("rmp_joint_accel_topic", "/rmp_joint_accel");
    declare_parameter("rmp_tcp_accel_topic", "/rmp_tcp_accel");
    declare_parameter("publish_joint_states", true);
    declare_parameter("joint_state_publish_topic", "joint_states");
    declare_parameter("publish_visualization", true);
    declare_parameter("publish_goal_marker", true);
    declare_parameter("publish_control_point_markers", true);
    declare_parameter("publish_body_obstacle_markers", true);
    declare_parameter("publish_end_effector_pose", true);
    declare_parameter("publish_debug_state", true);
    declare_parameter("publish_repulsion_metric_markers", true);
    declare_parameter("control_point_marker_diameter", 0.10);
    declare_parameter("repulsion_metric_marker_topic", "repulsion_metric_markers");
    declare_parameter("repulsion_metric_marker_min_norm", 0.01);
    declare_parameter("repulsion_metric_marker_dot_diameter", 0.04);
    declare_parameter("publish_tcp_accel_marker", true);
    declare_parameter("tcp_accel_marker_topic", "tcp_accel_marker");
    declare_parameter("tcp_accel_marker_max_length", 0.15);
    declare_parameter("tcp_accel_marker_norm_for_max_length", 2.0);
    declare_parameter("tcp_accel_marker_min_norm", 0.001);
	    declare_parameter("enable_tangent_escape_filter", false);
    declare_parameter("enable_tangent_escape_rmp", false);
    declare_parameter("tangent_escape_rmp_metric_scalar", 0.0);
    declare_parameter("tangent_escape_rmp_normal_metric_scalar", 0.0);
    declare_parameter("tangent_escape_rmp_damping_gain", 0.0);
    declare_parameter("tangent_escape_rmp_escape_speed", 0.05);
    declare_parameter("tangent_escape_rmp_velocity_gain", 5.0);
    declare_parameter("tangent_escape_rmp_max_accel", 0.8);
    declare_parameter("tangent_escape_rmp_goal_block_beta_on", 0.5);
    declare_parameter("tangent_escape_rmp_goal_block_beta_full", 0.94);
	    declare_parameter("tangent_escape_filter_safe_distance", 0.08);
    declare_parameter("tangent_escape_filter_influence_distance", 0.25);
    declare_parameter("tangent_escape_filter_tangent_gain", 2.0);
    declare_parameter("tangent_escape_filter_normal_gain", 0.0);
    declare_parameter("tangent_escape_filter_damping", 0.02);
    declare_parameter("tangent_escape_filter_max_delta_qdd", 8.0);
    declare_parameter("tangent_escape_filter_candidate_count", 16);
    declare_parameter("tangent_escape_filter_candidate_lookahead", 0.08);
    declare_parameter("tangent_escape_filter_goal_weight", 1.0);
    declare_parameter("tangent_escape_filter_continuity_weight", 0.5);
    declare_parameter("tangent_escape_filter_up_weight", 0.0);
    declare_parameter("tangent_escape_filter_duplicate_risk_weight", 2.0);
    declare_parameter("tangent_escape_filter_adjacent_block_weight", 1.0);
    declare_parameter("tangent_escape_filter_branch_hold_weight", 0.5);
    declare_parameter("tangent_escape_filter_min_activation", 0.05);
    declare_parameter("tangent_escape_filter_min_tangent_norm", 1e-4);
    declare_parameter("tangent_escape_filter_goal_normal_dot_threshold", 1.0);
    declare_parameter("publish_tangent_escape_filter_debug", false);
    declare_parameter("tangent_escape_filter_marker_topic", "tangent_escape_filter_markers");
    declare_parameter("tangent_escape_filter_marker_length", 0.12);
    declare_parameter("publish_tangent_escape_filter_data", false);
    declare_parameter("tangent_escape_filter_data_topic", "/tangent_escape_filter_data");
    declare_parameter("publish_tangent_escape_filter_candidate_data", false);
    declare_parameter(
      "tangent_escape_filter_candidate_data_topic",
      "/tangent_escape_filter_candidates");
    declare_parameter("publish_rmp_ee_pose", true);
    declare_parameter("publish_goal_tf", true);
    declare_parameter("goal_tf_parent_frame", "base_link");
    declare_parameter("goal_tf_child_frame", "rmp_goal_target");
    declare_parameter("servo_t1", 0.002);
    declare_parameter("servo_t2", 0.1);
    declare_parameter("servo_gain", 0.02);
    declare_parameter("servo_alpha", 0.2);
    declare_parameter("speedj_t1", 0.02);
    declare_parameter("speedj_t2", 0.2);
    declare_parameter("speedj_gain", 0.05);
    declare_parameter("speedj_alpha", 0.1);
    declare_parameter("startup_move_to_default_pose", true);
    declare_parameter(
      "startup_home_joints_deg",
      std::vector<double>{88.82315826, 1.57005262, -108.45492554, 16.88487434, -89.99609375, 1.24207485});
    declare_parameter("startup_movej_speed", 20.0);
    declare_parameter("startup_movej_accel", 20.0);
    declare_parameter("startup_release_tolerance_deg", 2.0);
    declare_parameter("startup_release_timeout_sec", 12.0);
    declare_parameter("stop_on_shutdown", true);
    declare_parameter("shutdown_action", "halt");
    declare_parameter("enable_socket_realtime", false);
    declare_parameter("socket_realtime_priority", 60);
    declare_parameter("enable_realtime", false);
    declare_parameter("realtime_priority", 80);
    declare_parameter("lock_memory", false);
    declare_parameter("graph.node_names", default_rmp_graph_node_names());
    declare_parameter("root_solve_offset", 1e-3);
    declare_parameter("solve_method", "rmp2");
    declare_parameter("rmp_type", "canonical");
    declare_parameter("pinocchio_urdf_path", "");
    declare_parameter(
      "external_rmp.topic_prefix",
      std::string("external_rmp"));
    declare_parameter(
      "body_obstacles.names",
      rclcpp::ParameterValue(std::vector<std::string>{}));
    declare_parameter("cspace_target_metric_scalar", 0.005);
    declare_parameter("cspace_target_position_gain", 100.0);
    declare_parameter("cspace_target_damping_gain", 50.0);
    declare_parameter("cspace_target_robust_position_term_thresh", 0.5);
    declare_parameter("cspace_target_inertia", 0.0001);
    declare_parameter("joint_limit_metric_scalar", 0.1);
    declare_parameter("joint_limit_metric_length_scale", 0.01);
    declare_parameter("joint_limit_metric_exploder_eps", 0.001);
    declare_parameter("joint_limit_metric_velocity_gate_length_scale", 0.01);
    declare_parameter("joint_limit_accel_damper_gain", 200.0);
    declare_parameter("joint_limit_accel_potential_gain", 1.0);
    declare_parameter("joint_limit_accel_potential_exploder_eps", 0.01);
    declare_parameter("joint_limit_accel_potential_exploder_length_scale", 0.1);
    declare_parameter("joint_velocity_cap_max_velocity", 1.7);
    declare_parameter("joint_velocity_cap_velocity_damping_region", 0.15);
    declare_parameter("joint_velocity_cap_damping_gain", 5.0);
    declare_parameter("joint_velocity_cap_metric_weight", 0.05);
    declare_parameter("max_joint_accel", 20.0);
    declare_parameter("measured_position_feedback_blend", 1.0);
    declare_parameter("measured_velocity_feedback_blend", 0.35);
    declare_parameter("estimate_velocity_in_controller", false);
    declare_parameter("controller_velocity_filter_alpha", 0.25);
    declare_parameter("use_velocity_feedback_in_solver", true);
    declare_parameter("target_rmp_accel_p_gain", 50.0);
    declare_parameter("target_rmp_accel_d_gain", 70.0);
    declare_parameter("command_guard_max_step_rad", 0.00436332313);
    declare_parameter("command_guard_max_velocity_rad_s", 0.436332313);
    declare_parameter("predictive_joint_limit_guard", true);
    declare_parameter("target_rmp_accel_norm_eps", 0.075);
    declare_parameter("target_rmp_metric_alpha_length_scale", 0.05);
    declare_parameter("target_rmp_min_metric_alpha", 0.03);
    declare_parameter("target_rmp_max_metric_scalar", 1.0);
    declare_parameter("target_rmp_min_metric_scalar", 0.5);
    declare_parameter("target_rmp_proximity_metric_boost_scalar", 3.0);
    declare_parameter("target_rmp_proximity_metric_boost_length_scale", 0.02);
    declare_parameter("axis_target_rmp_accel_p_gain", 1000.0);
    declare_parameter("axis_target_rmp_accel_d_gain", 500.0);
    declare_parameter("axis_target_rmp_metric_scalar", 50.0);
    declare_parameter("axis_target_rmp_proximity_metric_boost_scalar", 10.0);
    declare_parameter("axis_target_rmp_proximity_metric_boost_length_scale", 0.1);
    declare_parameter("wrist_axis_target_rmp_accel_p_gain", 50.0);
    declare_parameter("wrist_axis_target_rmp_accel_d_gain", 50.0);
    declare_parameter("wrist_axis_target_rmp_metric_scalar", 1000.0);
    declare_parameter("wrist_axis_target_rmp_proximity_metric_boost_scalar", 1.0);
    declare_parameter("wrist_axis_target_rmp_proximity_metric_boost_length_scale", 0.01);
    declare_parameter("collision_policy", "repulsive");
    declare_parameter("collision_rmp_margin", 0.0);
    declare_parameter("collision_rmp_damping_gain", 50.0);
    declare_parameter("collision_rmp_damping_std_dev", 0.04);
    declare_parameter("collision_rmp_damping_robustness_eps", 0.01);
    declare_parameter("collision_rmp_damping_velocity_gate_length_scale", 0.01);
    declare_parameter("collision_rmp_repulsion_gain", 800.0);
    declare_parameter("collision_rmp_repulsion_std_dev", 0.01);
    declare_parameter("collision_rmp_metric_modulation_radius", 0.5);
    declare_parameter("collision_rmp_metric_scalar", 1.0);
    declare_parameter("collision_rmp_metric_exploder_std_dev", 0.02);
    declare_parameter("collision_rmp_metric_exploder_eps", 0.001);
    declare_parameter("damping_rmp_accel_d_gain", 30.0);
    declare_parameter("damping_rmp_metric_scalar", 0.005);
    declare_parameter("damping_rmp_inertia", 0.3);

    const auto initial_q = get_parameter("initial_q").as_double_array();
    state_.q = JointVector::Zero();
    state_.qd = JointVector::Zero();
    for (std::size_t index = 0; index < std::min<std::size_t>(initial_q.size(), 6); ++index) {
      state_.q[static_cast<int>(index)] = initial_q[index];
    }
    const auto initial_context = RB10Model::forward_context(state_.q);
    GoalTarget initial_goal_target;
    initial_goal_target.position = Eigen::Vector3d(
      get_parameter("goal_x").as_double(),
      get_parameter("goal_y").as_double(),
      get_parameter("goal_z").as_double());
    initialize_goal_from_first_state_ =
      get_parameter("initialize_goal_from_first_state").as_bool();
    min_goal_z_ = get_parameter("min_goal_z").as_double();
    safety_stop_on_min_z_ = get_parameter("safety_stop_on_min_z").as_bool();
    workspace_floor_z_ = get_parameter("workspace_floor_z").as_double();
    min_link6_z_ = get_parameter("min_link6_z").as_double();
    min_tcp_z_ = get_parameter("min_tcp_z").as_double();
    control_rate_hz_ = get_parameter("control_rate").as_double();
    enable_realtime_ = get_parameter("enable_realtime").as_bool();
    realtime_priority_ = get_parameter("realtime_priority").as_int();
    lock_memory_ = get_parameter("lock_memory").as_bool();
    const bool use_synced_input_velocity_filter =
      get_parameter("use_synced_input_velocity_filter").as_bool();
    estimate_velocity_in_controller_ =
      get_parameter("estimate_velocity_in_controller").as_bool();
    measured_position_feedback_blend_ = std::clamp(
      get_parameter("measured_position_feedback_blend").as_double(), 0.0, 1.0);
    measured_velocity_feedback_blend_ = std::clamp(
      get_parameter("measured_velocity_feedback_blend").as_double(), 0.0, 1.0);
    use_velocity_feedback_in_solver_ =
      get_parameter("use_velocity_feedback_in_solver").as_bool();
    controller_velocity_filter_alpha_ = std::clamp(
      get_parameter("controller_velocity_filter_alpha").as_double(), 0.0, 1.0);
    max_joint_accel_ = get_parameter("max_joint_accel").as_double();
    command_mode_ = parse_command_mode(get_parameter("command_mode").as_string());
    command_guard_max_step_rad_ = std::max(
      get_parameter("command_guard_max_step_rad").as_double(),
      1e-4);
    const double configured_max_velocity =
      get_parameter("command_guard_max_velocity_rad_s").as_double();
    command_guard_max_velocity_rad_s_ = std::max(
      configured_max_velocity,
      command_mode_ == CommandMode::kVelocity ? 1e-4 : command_guard_max_step_rad_ * control_rate_hz_);
    predictive_joint_limit_guard_ = get_parameter("predictive_joint_limit_guard").as_bool();
    joint_lower_limits_ = parse_joint_limit_degrees(
      "joint_lower_limits_deg",
      RB10Model::joint_lower_limits);
    joint_upper_limits_ = parse_joint_limit_degrees(
      "joint_upper_limits_deg",
      RB10Model::joint_upper_limits);
    for (std::size_t index = 0; index < joint_lower_limits_.size(); ++index) {
      if (joint_lower_limits_[index] >= joint_upper_limits_[index]) {
        throw std::runtime_error(
                "joint_lower_limits_deg must be less than joint_upper_limits_deg for every joint");
      }
    }
    const auto joint_limit_buffers = get_parameter("joint_limit_buffers").as_double_array();
    for (std::size_t index = 0; index < std::min<std::size_t>(joint_limit_buffers.size(), 6); ++index) {
      joint_limit_buffers_[index] = std::max(0.0, joint_limit_buffers[index]);
    }
    if (use_synced_input_velocity_filter && estimate_velocity_in_controller_) {
      RCLCPP_WARN(
        get_logger(),
        "use_synced_input_velocity_filter is enabled but estimate_velocity_in_controller is also true; the controller-side finite-difference estimator will override the synced backend velocity unless you set estimate_velocity_in_controller:=false");
    }
    body_goal_ = parse_vector3_parameter("body_goal", Eigen::Vector3d(0.45, 0.0, 0.9));
    initial_goal_target.orientation =
      Eigen::Quaterniond(initial_context.link_rotations[RB10Model::TCP_RMP]);
    initial_goal_target.orientation.normalize();
    goal_target_box_.set(initial_goal_target);
    declare_graph_parameters();
    declare_body_obstacle_parameters();
    obstacles_box_.set(std::vector<ObstacleSphere>{ObstacleSphere{}});
    proximity_sensor_obstacles_box_.set(
      std::vector<std::optional<ObstacleSphere>>(RB10Model::sensor_control_points.size()));
    tcp_accel_visualization_box_.set(TcpAccelerationSample{});
    tangent_escape_debug_box_.set(TangentEscapeDebugState{});
    const auto solver_config = build_solver_config();
    target_metric_params_ = solver_config.target;
    collision_metric_params_ = solver_config.collision;
    configure_external_rmp_inputs(solver_config.graph_nodes);
    solver_ = build_solver(solver_config);
    body_obstacles_visual_ = solver_config.body_obstacles;

    publish_joint_states_enabled_ = get_parameter("publish_joint_states").as_bool();
    publish_visualization_enabled_ = get_parameter("publish_visualization").as_bool();
    publish_goal_marker_enabled_ =
      get_parameter("publish_goal_marker").as_bool();
    publish_control_point_markers_enabled_ =
      get_parameter("publish_control_point_markers").as_bool();
    publish_body_obstacle_markers_enabled_ =
      get_parameter("publish_body_obstacle_markers").as_bool();
    publish_end_effector_pose_enabled_ =
      get_parameter("publish_end_effector_pose").as_bool();
    publish_debug_state_enabled_ =
      get_parameter("publish_debug_state").as_bool();
    publish_repulsion_metric_markers_enabled_ =
      get_parameter("publish_repulsion_metric_markers").as_bool();
    control_point_marker_diameter_ =
      get_parameter("control_point_marker_diameter").as_double();
    repulsion_metric_marker_min_norm_ = std::clamp(
      get_parameter("repulsion_metric_marker_min_norm").as_double(), 0.0, 1.0);
    repulsion_metric_marker_dot_diameter_ = std::max(
      get_parameter("repulsion_metric_marker_dot_diameter").as_double(), 0.005);
    publish_tcp_accel_marker_enabled_ =
      get_parameter("publish_tcp_accel_marker").as_bool();
    tcp_accel_marker_max_length_ = std::max(
      get_parameter("tcp_accel_marker_max_length").as_double(), 0.01);
    tcp_accel_marker_norm_for_max_length_ = std::max(
      get_parameter("tcp_accel_marker_norm_for_max_length").as_double(), 1e-9);
    tcp_accel_marker_min_norm_ = std::max(
      get_parameter("tcp_accel_marker_min_norm").as_double(), 0.0);
    enable_tangent_escape_filter_ =
      get_parameter("enable_tangent_escape_filter").as_bool();
    tangent_escape_safe_distance_ = std::max(
      get_parameter("tangent_escape_filter_safe_distance").as_double(), 0.0);
    tangent_escape_influence_distance_ = std::max(
      get_parameter("tangent_escape_filter_influence_distance").as_double(),
      tangent_escape_safe_distance_ + 1e-6);
    tangent_escape_tangent_gain_ = std::max(
      get_parameter("tangent_escape_filter_tangent_gain").as_double(), 0.0);
    tangent_escape_normal_gain_ = std::max(
      get_parameter("tangent_escape_filter_normal_gain").as_double(), 0.0);
    tangent_escape_damping_ = std::max(
      get_parameter("tangent_escape_filter_damping").as_double(), 1e-6);
    tangent_escape_max_delta_qdd_ = std::max(
      get_parameter("tangent_escape_filter_max_delta_qdd").as_double(), 0.0);
    tangent_escape_candidate_count_ = std::max(
      static_cast<int>(get_parameter("tangent_escape_filter_candidate_count").as_int()), 4);
    tangent_escape_candidate_lookahead_ = std::max(
      get_parameter("tangent_escape_filter_candidate_lookahead").as_double(), 1e-4);
    tangent_escape_goal_weight_ =
      get_parameter("tangent_escape_filter_goal_weight").as_double();
    tangent_escape_continuity_weight_ =
      get_parameter("tangent_escape_filter_continuity_weight").as_double();
    tangent_escape_up_weight_ =
      get_parameter("tangent_escape_filter_up_weight").as_double();
    tangent_escape_duplicate_risk_weight_ =
      get_parameter("tangent_escape_filter_duplicate_risk_weight").as_double();
    tangent_escape_adjacent_block_weight_ =
      get_parameter("tangent_escape_filter_adjacent_block_weight").as_double();
    tangent_escape_branch_hold_weight_ =
      get_parameter("tangent_escape_filter_branch_hold_weight").as_double();
    tangent_escape_min_activation_ = std::clamp(
      get_parameter("tangent_escape_filter_min_activation").as_double(), 0.0, 1.0);
    tangent_escape_min_tangent_norm_ = std::max(
      get_parameter("tangent_escape_filter_min_tangent_norm").as_double(), 1e-9);
    tangent_escape_goal_normal_dot_threshold_ =
      get_parameter("tangent_escape_filter_goal_normal_dot_threshold").as_double();
    publish_tangent_escape_filter_debug_enabled_ =
      get_parameter("publish_tangent_escape_filter_debug").as_bool();
    tangent_escape_marker_length_ = std::max(
      get_parameter("tangent_escape_filter_marker_length").as_double(), 0.01);
    publish_tangent_escape_filter_data_enabled_ =
      get_parameter("publish_tangent_escape_filter_data").as_bool();
    publish_tangent_escape_filter_candidate_data_enabled_ =
      get_parameter("publish_tangent_escape_filter_candidate_data").as_bool();
    publish_rmp_ee_pose_enabled_ = get_parameter("publish_rmp_ee_pose").as_bool();
    publish_goal_tf_enabled_ = get_parameter("publish_goal_tf").as_bool();
    goal_tf_parent_frame_ = get_parameter("goal_tf_parent_frame").as_string();
    goal_tf_child_frame_ = get_parameter("goal_tf_child_frame").as_string();
    joint_state_topic_ = get_parameter("joint_state_topic").as_string();
    rmp_flag_gate_enabled_ = get_parameter("rmp_flag_gate_enabled").as_bool();
    rmp_active_flag_value_ = static_cast<int>(get_parameter("rmp_active_flag_value").as_int());
    rmp_active_.store(!rmp_flag_gate_enabled_);
    if (publish_joint_states_enabled_) {
      joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(
        get_parameter("joint_state_publish_topic").as_string(),
        10);
      rt_joint_state_pub_ = std::make_shared<JointStateRtPublisher>(joint_state_pub_);
    }
    if (get_parameter("backend_mode").as_string() == "rb10_direct_api") {
      direct_command_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("position_command_topic").as_string(), 10);
      rt_direct_command_pub_ = std::make_shared<Float64ArrayRtPublisher>(direct_command_pub_);
    }
    if (get_parameter("publish_position_command").as_bool()) {
      position_command_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("position_command_state_topic").as_string(), 10);
      rt_position_command_pub_ =
        std::make_shared<Float64ArrayRtPublisher>(position_command_pub_);
    }
    if (get_parameter("publish_target_q").as_bool()) {
      target_q_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("target_q_topic").as_string(), 10);
      rt_target_q_pub_ = std::make_shared<Float64ArrayRtPublisher>(target_q_pub_);
    }
    if (get_parameter("publish_target_metric").as_bool()) {
      target_metric_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("target_metric_topic").as_string(), 10);
      rt_target_metric_pub_ = std::make_shared<Float64ArrayRtPublisher>(target_metric_pub_);
    }
    if (get_parameter("publish_rmp_accel_debug").as_bool()) {
      rmp_joint_accel_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("rmp_joint_accel_topic").as_string(), 10);
      rt_rmp_joint_accel_pub_ = std::make_shared<Float64ArrayRtPublisher>(rmp_joint_accel_pub_);
      rmp_tcp_accel_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("rmp_tcp_accel_topic").as_string(), 10);
      rt_rmp_tcp_accel_pub_ = std::make_shared<Float64ArrayRtPublisher>(rmp_tcp_accel_pub_);
    }
    if (publish_tangent_escape_filter_data_enabled_) {
      tangent_escape_filter_data_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        get_parameter("tangent_escape_filter_data_topic").as_string(), 10);
      rt_tangent_escape_filter_data_pub_ =
        std::make_shared<Float64ArrayRtPublisher>(tangent_escape_filter_data_pub_);
    }
    if (publish_tangent_escape_filter_candidate_data_enabled_) {
      tangent_escape_filter_candidate_data_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>(
          get_parameter("tangent_escape_filter_candidate_data_topic").as_string(), 10);
      rt_tangent_escape_filter_candidate_data_pub_ =
        std::make_shared<Float64ArrayRtPublisher>(tangent_escape_filter_candidate_data_pub_);
    }
    if (publish_visualization_enabled_) {
      if (publish_goal_marker_enabled_) {
        goal_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("goal_marker", 10);
      }
      if (publish_control_point_markers_enabled_) {
        control_point_pub_ =
          create_publisher<visualization_msgs::msg::MarkerArray>("control_points", 10);
      }
      if (publish_body_obstacle_markers_enabled_) {
        body_obstacle_pub_ =
          create_publisher<visualization_msgs::msg::MarkerArray>("body_obstacle_markers", 10);
      }
      if (publish_repulsion_metric_markers_enabled_) {
        repulsion_metric_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
          get_parameter("repulsion_metric_marker_topic").as_string(), 10);
      }
      if (publish_tcp_accel_marker_enabled_) {
        tcp_accel_pub_ = create_publisher<visualization_msgs::msg::Marker>(
          get_parameter("tcp_accel_marker_topic").as_string(), 10);
      }
      if (publish_tangent_escape_filter_debug_enabled_) {
        tangent_escape_debug_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
          get_parameter("tangent_escape_filter_marker_topic").as_string(), 10);
      }
      if (publish_end_effector_pose_enabled_) {
        eef_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
          "end_effector_pose",
          10);
      }
      if (publish_debug_state_enabled_) {
        debug_state_pub_ =
          create_publisher<std_msgs::msg::Float64MultiArray>("rmp_debug_state", 10);
      }
    }
    if (publish_rmp_ee_pose_enabled_) {
      rmp_eef_pose_pub_ = create_publisher<geometry_msgs::msg::Pose>("rmp_ee_pose", 10);
      rt_rmp_eef_pose_pub_ = std::make_shared<PoseRtPublisher>(rmp_eef_pose_pub_);
    }
    if (publish_goal_tf_enabled_) {
      goal_tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    goal_sub_ = create_subscription<geometry_msgs::msg::Point>(
      "goal_position",
      10,
      std::bind(&RmpflowControllerNode::on_goal, this, std::placeholders::_1));
    goal_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "goal_pose",
      10,
      std::bind(&RmpflowControllerNode::on_goal_pose, this, std::placeholders::_1));
    if (rmp_flag_gate_enabled_) {
      rmp_flag_sub_ = create_subscription<std_msgs::msg::UInt8>(
        get_parameter("rmp_flag_topic").as_string(),
        10,
        std::bind(&RmpflowControllerNode::on_rmp_flag, this, std::placeholders::_1));
    }
    obstacle_sub_ = create_subscription<visualization_msgs::msg::MarkerArray>(
      "obstacles",
      10,
      std::bind(&RmpflowControllerNode::on_obstacles, this, std::placeholders::_1));

    const auto backend_mode = get_parameter("backend_mode").as_string();
    if (backend_mode == "hardware_bridge") {
      backend_ = std::make_unique<HardwareBridgeBackend>(
        this,
        state_.q,
        get_parameter("hardware_state_topic").as_string(),
        get_parameter("hardware_command_topic").as_string());
      RCLCPP_INFO(get_logger(), "Using hardware bridge backend");
    } else if (backend_mode == "rb10_direct_api") {
      backend_ = std::make_unique<DirectRb10Backend>(
        this,
        state_.q,
        get_parameter("robot_ip").as_string(),
        get_parameter("simulation_mode").as_bool(),
        get_parameter("real_joint_state_source").as_string(),
        get_parameter("publish_debug_joint_state_sources").as_bool(),
        command_mode_,
        get_parameter("hardware_data_request_rate").as_double(),
        get_parameter("control_rate").as_double(),
        get_parameter("servo_t1").as_double(),
        get_parameter("servo_t2").as_double(),
        get_parameter("servo_gain").as_double(),
        get_parameter("servo_alpha").as_double(),
        get_parameter("speedj_t1").as_double(),
        get_parameter("speedj_t2").as_double(),
        get_parameter("speedj_gain").as_double(),
        get_parameter("speedj_alpha").as_double(),
        get_parameter("use_synced_input_velocity_filter").as_bool(),
        get_parameter("synced_input_velocity_filter_alpha").as_double(),
        get_parameter("synced_input_velocity_filter_beta").as_double(),
        get_parameter("synced_input_velocity_filter_type").as_string(),
        get_parameter("synced_input_velocity_ratio_tolerance").as_double(),
        get_parameter("startup_move_to_default_pose").as_bool(),
        get_parameter("startup_home_joints_deg").as_double_array(),
        get_parameter("startup_movej_speed").as_double(),
        get_parameter("startup_movej_accel").as_double(),
        get_parameter("startup_release_tolerance_deg").as_double(),
        get_parameter("startup_release_timeout_sec").as_double(),
        get_parameter("enable_socket_realtime").as_bool(),
        get_parameter("socket_realtime_priority").as_int(),
        get_parameter("stop_on_shutdown").as_bool(),
        get_parameter("shutdown_action").as_string());
      RCLCPP_INFO(
        get_logger(),
        "Using direct RB10 API backend (%s, state source: %s, command_mode=%s)",
        get_parameter("robot_ip").as_string().c_str(),
        get_parameter("real_joint_state_source").as_string().c_str(),
        command_mode_to_string(command_mode_));
    } else if (backend_mode == "joint_command_topics") {
      backend_ = std::make_unique<JointCommandTopicsBackend>(
        this,
        state_.q,
        get_parameter("joint_state_topic").as_string(),
        get_parameter("position_command_topic").as_string(),
        command_mode_);
      RCLCPP_INFO(
        get_logger(),
        "Using joint command topic backend (%s -> %s, command_mode=%s)",
        get_parameter("joint_state_topic").as_string().c_str(),
        get_parameter("position_command_topic").as_string().c_str(),
        command_mode_to_string(command_mode_));
    } else {
      backend_ = std::make_unique<SimulationBackend>(state_.q);
      RCLCPP_INFO(get_logger(), "Using simulation backend");
    }

    if (publish_joint_states_enabled_ && backend_->ready()) {
      publish_joint_states(backend_->read_state());
    }

    if (publish_visualization_enabled_ || publish_goal_tf_enabled_) {
      const double visualization_rate = get_parameter("visualization_rate").as_double();
      if (visualization_rate > std::numeric_limits<double>::epsilon()) {
        const auto visualization_period = std::chrono::duration<double>(1.0 / visualization_rate);
        visualization_timer_ = create_wall_timer(
          std::chrono::duration_cast<std::chrono::milliseconds>(visualization_period),
          std::bind(&RmpflowControllerNode::publish_visualization, this));
      } else {
        RCLCPP_WARN(
          get_logger(),
          "publish_visualization is enabled but visualization_rate <= 0.0; skipping visualization timer");
      }
    }

    control_thread_ = std::thread(&RmpflowControllerNode::control_loop, this);

    RCLCPP_INFO(
      get_logger(),
      "C++ RMPflow controller started at %.1f Hz",
      get_parameter("control_rate").as_double());
  }

  ~RmpflowControllerNode() override
  {
    running_.store(false);
    if (control_thread_.joinable()) {
      control_thread_.join();
    }
  }

private:
  void on_rmp_flag(const std_msgs::msg::UInt8::SharedPtr msg)
  {
    const bool requested_active = static_cast<int>(msg->data) == rmp_active_flag_value_;
    const bool was_active = rmp_active_.exchange(requested_active);
    if (was_active == requested_active) {
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Received controller /RMP_flag: %d -> active=%s",
      static_cast<int>(msg->data),
      requested_active ? "true" : "false");
    if (!requested_active) {
      virtual_velocity_state_initialized_ = false;
      last_safe_command_state_initialized_ = false;
      last_min_z_safety_triggered_.store(false);
      visualization_cleared_for_inactive_ = false;
    }
  }

  void on_goal(const geometry_msgs::msg::Point::SharedPtr msg)
  {
    GoalTarget next_goal;
    goal_target_box_.get(next_goal);
    next_goal.position = Eigen::Vector3d(
      msg->x,
      msg->y,
      std::max(msg->z, min_goal_z_));
    goal_target_box_.set(next_goal);
    external_goal_received_.store(true);
    startup_goal_synced_.store(true);
  }

  void on_goal_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    GoalTarget next_goal;
    next_goal.position = Eigen::Vector3d(
      msg->pose.position.x,
      msg->pose.position.y,
      std::max(msg->pose.position.z, min_goal_z_));
    Eigen::Quaterniond goal_orientation(
      msg->pose.orientation.w,
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z);
    if (!goal_orientation.coeffs().allFinite() || goal_orientation.norm() < 1e-9) {
      next_goal.orientation = Eigen::Quaterniond::Identity();
    } else {
      next_goal.orientation = goal_orientation.normalized();
    }
    RCLCPP_INFO(
      get_logger(),
      "Received controller /goal_pose: position=(%.4f, %.4f, %.4f), orientation=(%.4f, %.4f, %.4f, %.4f)",
      msg->pose.position.x,
      msg->pose.position.y,
      msg->pose.position.z,
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z,
      msg->pose.orientation.w);
    goal_target_box_.set(next_goal);
    external_goal_received_.store(true);
    startup_goal_synced_.store(true);
  }

  void on_obstacles(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
  {
    std::vector<ObstacleSphere> obstacles;
    std::vector<std::optional<ObstacleSphere>> proximity_sensor_obstacles(
      RB10Model::sensor_control_points.size());
    for (const auto & marker : msg->markers) {
      const auto proximity_control_point_index = proximity_marker_control_point_index(marker);
      if (marker.action != visualization_msgs::msg::Marker::ADD) {
        continue;
      }
	      const Eigen::Vector3d marker_center(
	        marker.pose.position.x,
	        marker.pose.position.y,
	        marker.pose.position.z);
	      const double obstacle_radius = 0.5 * std::max(marker.scale.x, marker.scale.y);
	      ObstacleSphere obstacle{marker_center, obstacle_radius};
      if (proximity_control_point_index.has_value()) {
        obstacle.proximity_control_point_index =
          static_cast<int>(proximity_control_point_index.value());
      }

	      if (
	        marker.type == visualization_msgs::msg::Marker::CYLINDER &&
	        marker.scale.z > obstacle_radius * 2.0)
	      {
        Eigen::Quaterniond orientation(
          marker.pose.orientation.w,
          marker.pose.orientation.x,
          marker.pose.orientation.y,
          marker.pose.orientation.z);
        if (orientation.norm() <= 1e-9) {
          orientation = Eigen::Quaterniond::Identity();
        }
        const Eigen::Vector3d cylinder_axis =
          orientation.normalized().toRotationMatrix() * Eigen::Vector3d::UnitZ();
        const double height = std::max(marker.scale.z, 0.0);
        const double spacing = std::max(obstacle_radius, 0.02);
        const int sphere_count =
          std::clamp(static_cast<int>(std::ceil(height / spacing)) + 1, 2, 64);
	        const double step = height / static_cast<double>(sphere_count - 1);
	        for (int index = 0; index < sphere_count; ++index) {
	          const double offset = -0.5 * height + step * static_cast<double>(index);
          ObstacleSphere cylinder_obstacle{
            marker_center + offset * cylinder_axis,
            obstacle_radius};
          cylinder_obstacle.proximity_control_point_index =
            obstacle.proximity_control_point_index;
	          obstacles.push_back(cylinder_obstacle);
	        }
	      } else {
	        obstacles.push_back(obstacle);
	      }
      if (proximity_control_point_index.has_value()) {
        proximity_sensor_obstacles[proximity_control_point_index.value()] = obstacle;
      }
    }
    if (obstacles.empty()) {
      obstacles.push_back(ObstacleSphere{});
    }
    obstacles_box_.set(obstacles);
    proximity_sensor_obstacles_box_.set(proximity_sensor_obstacles);
  }

  std::optional<std::size_t> proximity_marker_control_point_index(
    const visualization_msgs::msg::Marker & marker) const
  {
    if (marker.ns != "proximity_obstacles") {
      return std::nullopt;
    }

    if (!marker.text.empty()) {
      for (std::size_t index = 0; index < RB10Model::sensor_control_points.size(); ++index) {
        if (marker.text == RB10Model::sensor_control_points[index].frame_name) {
          return index;
        }
      }
      return std::nullopt;
    }

    return std::nullopt;
  }

  EigenRmpConfig build_solver_config() const
  {
    EigenRmpConfig config;
    config.graph_nodes = parse_graph_nodes();
    config.solve_offset = get_parameter("root_solve_offset").as_double();
    config.solve_method = get_parameter("solve_method").as_string();
    config.rmp_type = get_parameter("rmp_type").as_string();
    config.body_obstacles = parse_body_obstacles();

    const auto default_q = get_parameter("default_q").as_double_array();
    for (std::size_t index = 0; index < std::min<std::size_t>(default_q.size(), 6); ++index) {
      config.default_q[index] = default_q[index];
    }
    config.joint_lower_limits = joint_lower_limits_;
    config.joint_upper_limits = joint_upper_limits_;

    const auto joint_limit_buffers = get_parameter("joint_limit_buffers").as_double_array();
    for (std::size_t index = 0; index < std::min<std::size_t>(joint_limit_buffers.size(), 6); ++index) {
      config.joint_limit_buffers[index] = joint_limit_buffers[index];
    }

    config.cspace_target.metric_scalar = get_parameter("cspace_target_metric_scalar").as_double();
    config.cspace_target.position_gain = get_parameter("cspace_target_position_gain").as_double();
    config.cspace_target.damping_gain = get_parameter("cspace_target_damping_gain").as_double();
    config.cspace_target.robust_position_term_thresh =
      get_parameter("cspace_target_robust_position_term_thresh").as_double();
    config.cspace_target.inertia = get_parameter("cspace_target_inertia").as_double();

    config.joint_limit.metric_scalar = get_parameter("joint_limit_metric_scalar").as_double();
    config.joint_limit.metric_length_scale =
      get_parameter("joint_limit_metric_length_scale").as_double();
    config.joint_limit.metric_exploder_eps =
      get_parameter("joint_limit_metric_exploder_eps").as_double();
    config.joint_limit.metric_velocity_gate_length_scale =
      get_parameter("joint_limit_metric_velocity_gate_length_scale").as_double();
    config.joint_limit.accel_damper_gain =
      get_parameter("joint_limit_accel_damper_gain").as_double();
    config.joint_limit.accel_potential_gain =
      get_parameter("joint_limit_accel_potential_gain").as_double();
    config.joint_limit.accel_potential_exploder_eps =
      get_parameter("joint_limit_accel_potential_exploder_eps").as_double();
    config.joint_limit.accel_potential_exploder_length_scale =
      get_parameter("joint_limit_accel_potential_exploder_length_scale").as_double();

    config.joint_velocity_cap.max_velocity =
      get_parameter("joint_velocity_cap_max_velocity").as_double();
    config.joint_velocity_cap.velocity_damping_region =
      get_parameter("joint_velocity_cap_velocity_damping_region").as_double();
    config.joint_velocity_cap.damping_gain =
      get_parameter("joint_velocity_cap_damping_gain").as_double();
    config.joint_velocity_cap.metric_weight =
      get_parameter("joint_velocity_cap_metric_weight").as_double();

    config.target.accel_p_gain = get_parameter("target_rmp_accel_p_gain").as_double();
    config.target.accel_d_gain = get_parameter("target_rmp_accel_d_gain").as_double();
    config.target.accel_norm_eps = get_parameter("target_rmp_accel_norm_eps").as_double();
    config.target.metric_alpha_length_scale =
      get_parameter("target_rmp_metric_alpha_length_scale").as_double();
    config.target.min_metric_alpha =
      get_parameter("target_rmp_min_metric_alpha").as_double();
    config.target.max_metric_scalar =
      get_parameter("target_rmp_max_metric_scalar").as_double();
    config.target.min_metric_scalar =
      get_parameter("target_rmp_min_metric_scalar").as_double();
    config.target.proximity_metric_boost_scalar =
      get_parameter("target_rmp_proximity_metric_boost_scalar").as_double();
    config.target.proximity_metric_boost_length_scale =
      get_parameter("target_rmp_proximity_metric_boost_length_scale").as_double();
    config.axis_target.accel_p_gain =
      get_parameter("axis_target_rmp_accel_p_gain").as_double();
    config.axis_target.accel_d_gain =
      get_parameter("axis_target_rmp_accel_d_gain").as_double();
    config.axis_target.metric_scalar =
      get_parameter("axis_target_rmp_metric_scalar").as_double();
    config.axis_target.proximity_metric_boost_scalar =
      get_parameter("axis_target_rmp_proximity_metric_boost_scalar").as_double();
    config.axis_target.proximity_metric_boost_length_scale =
      get_parameter("axis_target_rmp_proximity_metric_boost_length_scale").as_double();
    config.wrist_axis_target.accel_p_gain =
      get_parameter("wrist_axis_target_rmp_accel_p_gain").as_double();
    config.wrist_axis_target.accel_d_gain =
      get_parameter("wrist_axis_target_rmp_accel_d_gain").as_double();
    config.wrist_axis_target.metric_scalar =
      get_parameter("wrist_axis_target_rmp_metric_scalar").as_double();
    config.wrist_axis_target.proximity_metric_boost_scalar =
      get_parameter("wrist_axis_target_rmp_proximity_metric_boost_scalar").as_double();
    config.wrist_axis_target.proximity_metric_boost_length_scale =
      get_parameter("wrist_axis_target_rmp_proximity_metric_boost_length_scale").as_double();

    config.collision.policy = get_parameter("collision_policy").as_string();
    config.collision.margin = get_parameter("collision_rmp_margin").as_double();
    config.collision.damping_gain = get_parameter("collision_rmp_damping_gain").as_double();
    config.collision.damping_std_dev = get_parameter("collision_rmp_damping_std_dev").as_double();
    config.collision.damping_robustness_eps =
      get_parameter("collision_rmp_damping_robustness_eps").as_double();
    config.collision.damping_velocity_gate_length_scale =
      get_parameter("collision_rmp_damping_velocity_gate_length_scale").as_double();
    config.collision.repulsion_gain =
      get_parameter("collision_rmp_repulsion_gain").as_double();
    config.collision.repulsion_std_dev =
      get_parameter("collision_rmp_repulsion_std_dev").as_double();
    config.collision.metric_modulation_radius =
      get_parameter("collision_rmp_metric_modulation_radius").as_double();
    config.collision.metric_scalar =
      get_parameter("collision_rmp_metric_scalar").as_double();
    config.collision.metric_exploder_std_dev =
      get_parameter("collision_rmp_metric_exploder_std_dev").as_double();
	    config.collision.metric_exploder_eps =
	      get_parameter("collision_rmp_metric_exploder_eps").as_double();

    config.tangent_escape.enabled = get_parameter("enable_tangent_escape_rmp").as_bool();
    config.tangent_escape.metric_scalar =
      get_parameter("tangent_escape_rmp_metric_scalar").as_double();
    config.tangent_escape.normal_metric_scalar =
      get_parameter("tangent_escape_rmp_normal_metric_scalar").as_double();
    config.tangent_escape.damping_gain =
      get_parameter("tangent_escape_rmp_damping_gain").as_double();
    config.tangent_escape.escape_speed =
      get_parameter("tangent_escape_rmp_escape_speed").as_double();
    config.tangent_escape.velocity_gain =
      get_parameter("tangent_escape_rmp_velocity_gain").as_double();
    config.tangent_escape.max_accel =
      get_parameter("tangent_escape_rmp_max_accel").as_double();
    config.tangent_escape.goal_block_beta_on =
      get_parameter("tangent_escape_rmp_goal_block_beta_on").as_double();
    config.tangent_escape.goal_block_beta_full =
      get_parameter("tangent_escape_rmp_goal_block_beta_full").as_double();
    config.tangent_escape.safe_distance =
      get_parameter("tangent_escape_filter_safe_distance").as_double();
    config.tangent_escape.influence_distance =
      get_parameter("tangent_escape_filter_influence_distance").as_double();
    config.tangent_escape.tangent_gain =
      get_parameter("tangent_escape_filter_tangent_gain").as_double();
    config.tangent_escape.normal_gain =
      get_parameter("tangent_escape_filter_normal_gain").as_double();
    config.tangent_escape.candidate_count =
      std::max(static_cast<int>(get_parameter("tangent_escape_filter_candidate_count").as_int()), 4);
    config.tangent_escape.candidate_lookahead =
      get_parameter("tangent_escape_filter_candidate_lookahead").as_double();
    config.tangent_escape.goal_weight =
      get_parameter("tangent_escape_filter_goal_weight").as_double();
    config.tangent_escape.continuity_weight =
      get_parameter("tangent_escape_filter_continuity_weight").as_double();
    config.tangent_escape.up_weight =
      get_parameter("tangent_escape_filter_up_weight").as_double();
    config.tangent_escape.duplicate_risk_weight =
      get_parameter("tangent_escape_filter_duplicate_risk_weight").as_double();
    config.tangent_escape.adjacent_block_weight =
      get_parameter("tangent_escape_filter_adjacent_block_weight").as_double();
    config.tangent_escape.branch_hold_weight =
      get_parameter("tangent_escape_filter_branch_hold_weight").as_double();
    config.tangent_escape.min_activation =
      get_parameter("tangent_escape_filter_min_activation").as_double();
    config.tangent_escape.min_tangent_norm =
      get_parameter("tangent_escape_filter_min_tangent_norm").as_double();
    config.tangent_escape.goal_normal_dot_threshold =
      get_parameter("tangent_escape_filter_goal_normal_dot_threshold").as_double();

	    config.damping.accel_d_gain = get_parameter("damping_rmp_accel_d_gain").as_double();
    config.damping.metric_scalar = get_parameter("damping_rmp_metric_scalar").as_double();
    config.damping.inertia = get_parameter("damping_rmp_inertia").as_double();
    return config;
  }

  std::unique_ptr<RmpSolverInterface> build_solver(const EigenRmpConfig & config) const
  {
    return std::make_unique<PinocchioDirectRmpSolver>(config, resolve_pinocchio_urdf_path());
  }

  std::string resolve_pinocchio_urdf_path() const
  {
    const auto configured = get_parameter("pinocchio_urdf_path").as_string();
    if (!configured.empty()) {
      return configured;
    }
    return ament_index_cpp::get_package_share_directory("rb10_rmpflow_rviz") +
           "/urdf/rb10_1300e.urdf";
  }

  void declare_graph_parameters()
  {
    const auto node_names = get_parameter("graph.node_names").as_string_array();
    const auto defaults = default_rmp_graph_nodes();
    for (const auto & name : node_names) {
      auto match_it = std::find_if(
        defaults.begin(),
        defaults.end(),
        [&name](const RmpNodeConfig & node) {return node.name == name;});
      const RmpNodeConfig fallback =
        match_it != defaults.end() ? *match_it : make_rmp_node_config(name, "root", "identity", "none", true);
      const std::string prefix = "graph." + name + ".";
      declare_parameter(prefix + "parent", fallback.parents.front());
      declare_parameter(prefix + "parents", fallback.parents);
      declare_parameter(prefix + "task_map", fallback.task_map_type);
      declare_parameter(prefix + "leaf", fallback.leaf_rmp_type);
      declare_parameter(prefix + "enabled", fallback.enabled);
      declare_parameter(prefix + "target_key", fallback.target_key);
      declare_parameter(prefix + "link_name", fallback.link_name);
      declare_parameter(prefix + "axis", fallback.axis);
      declare_parameter(prefix + "handcrafted_leaf", fallback.handcrafted_leaf_rmp_type);
      declare_parameter(prefix + "parent_weights", fallback.parent_weights);
      declare_parameter(prefix + "bias", fallback.bias);
      declare_parameter(prefix + "matrix", fallback.matrix);
      declare_parameter(prefix + "slice_start", fallback.slice_start);
      declare_parameter(prefix + "slice_length", fallback.slice_length);
      declare_parameter(prefix + "scale", fallback.scale);
      declare_parameter(prefix + "identity_multiplier", fallback.identity_multiplier);
      declare_parameter(prefix + "epsilon", fallback.epsilon);
    }
  }

  std::vector<RmpNodeConfig> parse_graph_nodes() const
  {
    const auto node_names = get_parameter("graph.node_names").as_string_array();
    std::vector<RmpNodeConfig> nodes;
    nodes.reserve(node_names.size());

    for (const auto & name : node_names) {
      const std::string prefix = "graph." + name + ".";
      auto parents = get_parameter(prefix + "parents").as_string_array();
      if (parents.empty()) {
        parents.push_back(get_parameter(prefix + "parent").as_string());
      }
      nodes.push_back(make_rmp_node_config(
        name,
        parents,
        get_parameter(prefix + "task_map").as_string(),
        get_parameter(prefix + "leaf").as_string(),
        get_parameter(prefix + "enabled").as_bool(),
        get_parameter(prefix + "target_key").as_string(),
        get_parameter(prefix + "link_name").as_string(),
        get_parameter(prefix + "axis").as_string(),
        get_parameter(prefix + "handcrafted_leaf").as_string(),
        get_parameter(prefix + "parent_weights").as_double_array(),
        get_parameter(prefix + "bias").as_double_array(),
        get_parameter(prefix + "matrix").as_double_array(),
        static_cast<int>(get_parameter(prefix + "slice_start").as_int()),
        static_cast<int>(get_parameter(prefix + "slice_length").as_int()),
        get_parameter(prefix + "scale").as_double(),
        get_parameter(prefix + "identity_multiplier").as_double(),
        get_parameter(prefix + "epsilon").as_double()));
    }

    return nodes;
  }

  std::vector<BodyObstacle> parse_body_obstacles() const
  {
    std::vector<std::string> names;
    get_parameter_or("body_obstacles.names", names, std::vector<std::string>{});
    std::vector<BodyObstacle> obstacles;
    obstacles.reserve(names.size());
    for (const auto & name : names) {
      const std::string prefix = "body_obstacles." + name + ".";
      BodyObstacle obstacle;
      obstacle.type = get_parameter(prefix + "type").as_string();
      obstacle.link_name = get_parameter(prefix + "link_name").as_string();
      obstacle.mins = parse_vector3_parameter(prefix + "mins", Eigen::Vector3d::Zero());
      obstacle.maxs = parse_vector3_parameter(prefix + "maxs", Eigen::Vector3d::Zero());
      obstacle.center = parse_vector3_parameter(prefix + "center", Eigen::Vector3d::Zero());
      obstacle.radius = get_parameter(prefix + "radius").as_double();
      obstacles.push_back(obstacle);
    }
    return obstacles;
  }

  int control_point_count() const
  {
    return static_cast<int>(RB10Model::sensor_control_points.size());
  }

  std::vector<std::size_t> build_topological_order(
    const std::vector<RmpNodeConfig> & nodes) const
  {
    std::unordered_map<std::string, std::size_t> enabled_nodes;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      if (!nodes[index].enabled) {
        continue;
      }
      enabled_nodes.emplace(nodes[index].name, index);
    }

    std::unordered_map<std::string, int> indegree;
    std::unordered_map<std::string, std::vector<std::string>> outgoing;
    for (const auto & entry : enabled_nodes) {
      indegree.emplace(entry.first, 0);
    }

    for (const auto & entry : enabled_nodes) {
      const auto & node = nodes[entry.second];
      for (const auto & parent : node.parents) {
        if (parent == "root") {
          continue;
        }
        if (!enabled_nodes.count(parent)) {
          throw std::runtime_error(
                  "Graph node " + node.name + " references missing parent " + parent);
        }
        ++indegree[node.name];
        outgoing[parent].push_back(node.name);
      }
    }

    std::vector<std::string> ready;
    for (const auto & entry : indegree) {
      if (entry.second == 0) {
        ready.push_back(entry.first);
      }
    }
    std::sort(ready.begin(), ready.end());

    std::vector<std::size_t> order;
    while (!ready.empty()) {
      const auto name = ready.front();
      ready.erase(ready.begin());
      order.push_back(enabled_nodes.at(name));
      for (const auto & child : outgoing[name]) {
        auto & child_indegree = indegree.at(child);
        --child_indegree;
        if (child_indegree == 0) {
          ready.push_back(child);
        }
      }
      std::sort(ready.begin(), ready.end());
    }

    if (order.size() != enabled_nodes.size()) {
      throw std::runtime_error("Cycle detected in graph configuration");
    }
    return order;
  }

  std::unordered_map<std::string, int> infer_node_dims(
    const std::vector<RmpNodeConfig> & nodes) const
  {
    std::unordered_map<std::string, int> dims;
    dims.emplace("root", 6);
    for (const auto index : build_topological_order(nodes)) {
      const auto & node = nodes[index];
      if (node.task_map_type == "tcp_position" ||
        node.task_map_type == "link_position" ||
        node.task_map_type == "link_orientation_axis")
      {
        dims[node.name] = 3;
      } else if (node.task_map_type == "joint_limit") {
        dims[node.name] = 12;
      } else if (node.task_map_type == "control_points") {
        dims[node.name] = 3 * control_point_count();
      } else if (node.task_map_type == "collision_distance") {
        dims[node.name] = control_point_count();
      } else if (node.task_map_type == "norm") {
        dims[node.name] = 1;
      } else if (node.task_map_type == "affine") {
        if (!node.bias.empty()) {
          dims[node.name] = static_cast<int>(node.bias.size());
        } else if (!node.matrix.empty()) {
          int input_dim = 0;
          for (const auto & parent : node.parents) {
            input_dim += dims.at(parent);
          }
          if (input_dim == 0 || static_cast<int>(node.matrix.size()) % input_dim != 0) {
            throw std::runtime_error("Invalid affine matrix size for node " + node.name);
          }
          dims[node.name] = static_cast<int>(node.matrix.size()) / input_dim;
        } else {
          dims[node.name] = dims.at(node.parents.front());
        }
      } else if (node.task_map_type == "concat") {
        int dim = 0;
        for (const auto & parent : node.parents) {
          dim += dims.at(parent);
        }
        dims[node.name] = dim;
      } else if (
        node.task_map_type == "elem_multiply" ||
        node.task_map_type == "elem_divide" ||
        node.task_map_type == "sin" ||
        node.task_map_type == "cos" ||
        node.task_map_type == "tanh" ||
        node.task_map_type == "square" ||
        node.task_map_type == "abs")
      {
        dims[node.name] = dims.at(node.parents.front());
      } else if (node.task_map_type == "slice") {
        dims[node.name] = node.slice_length > 0 ? node.slice_length :
          dims.at(node.parents.front()) - node.slice_start;
      } else {
        dims[node.name] = dims.at(node.parents.front());
      }
    }
    return dims;
  }

  void declare_external_rmp_feature_parameters(const std::vector<RmpNodeConfig> & nodes)
  {
    std::vector<std::string> feature_keys;
    for (const auto & node : nodes) {
      if (node.leaf_rmp_type == "external" || node.handcrafted_leaf_rmp_type == "external") {
        feature_keys.push_back(node.target_key);
      }
    }
    std::sort(feature_keys.begin(), feature_keys.end());
    feature_keys.erase(std::unique(feature_keys.begin(), feature_keys.end()), feature_keys.end());
    for (const auto & key : feature_keys) {
      const std::string prefix = "external_rmp." + key + ".";
      declare_parameter(prefix + "enabled", true);
      declare_parameter(prefix + "topic_prefix", key);
    }
  }

  void configure_external_rmp_inputs(const std::vector<RmpNodeConfig> & nodes)
  {
    declare_external_rmp_feature_parameters(nodes);
    const auto dims = infer_node_dims(nodes);
    const std::string root_prefix = get_parameter("external_rmp.topic_prefix").as_string();
    for (const auto & node : nodes) {
      const bool uses_external =
        node.leaf_rmp_type == "external" || node.handcrafted_leaf_rmp_type == "external";
      if (!uses_external) {
        continue;
      }
      const std::string key = node.target_key;
      const std::string prefix = "external_rmp." + key + ".";
      if (!get_parameter(prefix + "enabled").as_bool()) {
        continue;
      }
      if (external_rmp_buffers_.count(key)) {
        continue;
      }
      const int dim = dims.at(node.name);
      external_rmp_buffers_.emplace(
        key,
        ExternalRmpBuffer{
          dim,
          Eigen::MatrixXd::Zero(dim, dim),
          Eigen::VectorXd::Zero(dim),
          false,
          false
        });
      const std::string topic_prefix =
        root_prefix + "/" + get_parameter(prefix + "topic_prefix").as_string();
      external_metric_subs_.push_back(create_subscription<std_msgs::msg::Float64MultiArray>(
          topic_prefix + "/metric_sqrt",
          10,
          [this, key, dim](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
            if (static_cast<int>(msg->data.size()) != dim * dim) {
              RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "External RMP metric_sqrt size mismatch for %s: expected %d, got %zu",
                key.c_str(),
                dim * dim,
                msg->data.size());
              return;
            }
            Eigen::MatrixXd matrix(dim, dim);
            for (int row = 0; row < dim; ++row) {
              for (int col = 0; col < dim; ++col) {
                matrix(row, col) = msg->data[static_cast<std::size_t>(row * dim + col)];
              }
            }
            std::scoped_lock lock(external_rmp_mutex_);
            auto & buffer = external_rmp_buffers_.at(key);
            buffer.metric_sqrt = matrix;
            buffer.has_metric = true;
          }));
      external_accel_subs_.push_back(create_subscription<std_msgs::msg::Float64MultiArray>(
          topic_prefix + "/acceleration",
          10,
          [this, key, dim](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
            if (static_cast<int>(msg->data.size()) != dim) {
              RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "External RMP acceleration size mismatch for %s: expected %d, got %zu",
                key.c_str(),
                dim,
                msg->data.size());
              return;
            }
            Eigen::VectorXd acceleration(dim);
            for (int index = 0; index < dim; ++index) {
              acceleration[index] = msg->data[static_cast<std::size_t>(index)];
            }
            std::scoped_lock lock(external_rmp_mutex_);
            auto & buffer = external_rmp_buffers_.at(key);
            buffer.acceleration = acceleration;
            buffer.has_acceleration = true;
          }));
      RCLCPP_INFO(
        get_logger(),
        "External RMP input enabled for key '%s' on topics %s/metric_sqrt and %s/acceleration",
        key.c_str(),
        topic_prefix.c_str(),
        topic_prefix.c_str());
    }
  }

  void declare_body_obstacle_parameters()
  {
    std::vector<std::string> names;
    get_parameter_or("body_obstacles.names", names, std::vector<std::string>{});
    for (const auto & name : names) {
      const std::string prefix = "body_obstacles." + name + ".";
      declare_parameter(prefix + "type", "ball");
      declare_parameter(prefix + "link_name", "");
      declare_parameter(prefix + "mins", std::vector<double>{0.0, 0.0, 0.0});
      declare_parameter(prefix + "maxs", std::vector<double>{0.0, 0.0, 0.0});
      declare_parameter(prefix + "center", std::vector<double>{0.0, 0.0, 0.0});
      declare_parameter(prefix + "radius", 0.0);
    }
  }

  Eigen::Vector3d parse_vector3_parameter(
    const std::string & name,
    const Eigen::Vector3d & fallback) const
  {
    const auto values = get_parameter(name).as_double_array();
    Eigen::Vector3d out = fallback;
    for (std::size_t index = 0; index < std::min<std::size_t>(values.size(), 3); ++index) {
      out[static_cast<int>(index)] = values[index];
    }
    return out;
  }

  std::size_t link_index_from_name(const std::string & link_name) const
  {
    for (std::size_t index = 0; index < RB10Model::link_names.size(); ++index) {
      if (link_name == RB10Model::link_names[index]) {
        return index;
      }
    }
    throw std::runtime_error("Unknown RB10 link name: " + link_name);
  }

  Eigen::Vector3d body_obstacle_center_world(
    const BodyObstacle & obstacle,
    const KinematicsContext & context) const
  {
    if (!obstacle.link_name.empty()) {
      const auto link_index = link_index_from_name(obstacle.link_name);
      const auto & rotation = context.link_rotations[link_index];
      const auto & origin = context.link_positions[link_index];
      if (obstacle.type == "box") {
        return origin + rotation * (0.5 * (obstacle.mins + obstacle.maxs));
      }
      return origin + rotation * obstacle.center;
    }

    if (obstacle.type == "box") {
      return 0.5 * (obstacle.mins + obstacle.maxs);
    }
    return obstacle.center;
  }

  double body_obstacle_min_z_world(
    const BodyObstacle & obstacle,
    const KinematicsContext & context) const
  {
    if (obstacle.type == "ball") {
      return body_obstacle_center_world(obstacle, context).z() - obstacle.radius;
    }

    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    if (!obstacle.link_name.empty()) {
      const auto link_index = link_index_from_name(obstacle.link_name);
      rotation = context.link_rotations[link_index];
      origin = context.link_positions[link_index];
    }

    double min_z = std::numeric_limits<double>::infinity();
    for (int x_sel = 0; x_sel < 2; ++x_sel) {
      for (int y_sel = 0; y_sel < 2; ++y_sel) {
        for (int z_sel = 0; z_sel < 2; ++z_sel) {
          const Eigen::Vector3d local_corner(
            x_sel == 0 ? obstacle.mins.x() : obstacle.maxs.x(),
            y_sel == 0 ? obstacle.mins.y() : obstacle.maxs.y(),
            z_sel == 0 ? obstacle.mins.z() : obstacle.maxs.z());
          min_z = std::min(min_z, (origin + rotation * local_corner).z());
        }
      }
    }
    return min_z;
  }

  bool body_obstacle_interacts_with_sensor_control_point(
    std::size_t /*point_index*/,
    const BodyObstacle & obstacle) const
  {
    (void)obstacle;
    return false;
  }

  struct FloorSafetyMetrics
  {
    double min_link_z{std::numeric_limits<double>::infinity()};
    double min_joint_z{std::numeric_limits<double>::infinity()};
    double min_control_point_z{std::numeric_limits<double>::infinity()};
    double min_body_obstacle_z{std::numeric_limits<double>::infinity()};
  };

  FloorSafetyMetrics compute_floor_safety_metrics(const KinematicsContext & context) const
  {
    FloorSafetyMetrics metrics;
    for (const auto & position : context.link_positions) {
      metrics.min_link_z = std::min(metrics.min_link_z, position.z());
    }
    for (const auto & origin : context.joint_origins) {
      metrics.min_joint_z = std::min(metrics.min_joint_z, origin.z());
    }
    for (const auto & control_point : context.control_points) {
      metrics.min_control_point_z = std::min(
        metrics.min_control_point_z,
        control_point.position.z() - control_point.radius);
    }
    for (const auto & obstacle : body_obstacles_visual_) {
      metrics.min_body_obstacle_z = std::min(
        metrics.min_body_obstacle_z,
        body_obstacle_min_z_world(obstacle, context));
    }
    return metrics;
  }

  void publish_debug_state(const RobotState & current_state, const KinematicsContext & context)
  {
    if (!debug_state_pub_) {
      return;
    }

    Eigen::Vector3d goal;
    std::vector<ObstacleSphere> obstacles;
    GoalTarget goal_target;
    goal_target_box_.get(goal_target);
    goal = goal_target.position;
    obstacles_box_.get(obstacles);

    double min_external_clearance = std::numeric_limits<double>::infinity();
    for (const auto & control_point : context.control_points) {
      for (const auto & obstacle : obstacles) {
        const double clearance =
          (control_point.position - obstacle.center).norm() -
          (control_point.radius + obstacle.radius);
        min_external_clearance = std::min(min_external_clearance, clearance);
      }
    }

    double min_body_clearance = std::numeric_limits<double>::infinity();
    for (std::size_t point_index = 0; point_index < context.control_points.size(); ++point_index) {
      const auto & control_point = context.control_points[point_index];
      for (const auto & obstacle : body_obstacles_visual_) {
        if (!body_obstacle_interacts_with_sensor_control_point(point_index, obstacle)) {
          continue;
        }
        if (obstacle.type != "ball") {
          continue;
        }
        const auto center = body_obstacle_center_world(obstacle, context);
        const double clearance =
          (control_point.position - center).norm() -
          (control_point.radius + obstacle.radius);
        min_body_clearance = std::min(min_body_clearance, clearance);
      }
    }

    if (!std::isfinite(min_external_clearance)) {
      min_external_clearance = 1e6;
    }
    if (!std::isfinite(min_body_clearance)) {
      min_body_clearance = 1e6;
    }

    const auto floor_metrics = compute_floor_safety_metrics(context);

    std_msgs::msg::Float64MultiArray debug;
    debug.data = {
      (context.tcp_position - goal).norm(),
      context.tcp_position.z(),
      context.link_positions[RB10Model::LINK6].z(),
      min_external_clearance,
      min_body_clearance,
      current_state.qd.norm(),
      last_min_z_safety_triggered_.load() ? 1.0 : 0.0,
      floor_metrics.min_link_z,
      floor_metrics.min_joint_z,
      floor_metrics.min_control_point_z,
      floor_metrics.min_body_obstacle_z,
    };
    debug_state_pub_->publish(debug);
  }

  struct RepulsionMetricSample
  {
    Eigen::Vector3d control_point{Eigen::Vector3d::Zero()};
    Eigen::Vector3d direction{Eigen::Vector3d::UnitZ()};
    double clearance{0.0};
    double metric_norm{0.0};
    double effective_metric_norm{0.0};
    double repulsion_norm{0.0};
    double intensity_norm{0.0};
  };

  struct TangentEscapeDirectionScore
  {
    std::size_t index{0};
    Eigen::Vector3d direction{Eigen::Vector3d::UnitX()};
    double goal_score{0.0};
    double continuity_score{0.0};
    double up_score{0.0};
    double duplicate_risk_score{0.0};
    double adjacent_block_score{0.0};
    double branch_hold_score{0.0};
    double total_score{0.0};
    bool selected{false};
  };

  struct TangentEscapeCandidate
  {
    std::size_t control_point_index{0};
    Eigen::Vector3d control_point{Eigen::Vector3d::Zero()};
    Eigen::Vector3d obstacle_center{Eigen::Vector3d::Zero()};
    Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
    Eigen::Vector3d tangent{Eigen::Vector3d::UnitX()};
    Eigen::Vector3d raw_accel{Eigen::Vector3d::Zero()};
    Eigen::Vector3d curvature{Eigen::Vector3d::Zero()};
    Eigen::Matrix<double, 3, 6> jacobian{Eigen::Matrix<double, 3, 6>::Zero()};
    double clearance{0.0};
    double activation{0.0};
    double score{0.0};
    bool has_tangent{false};
    std::size_t selected_direction_index{0};
    std::vector<TangentEscapeDirectionScore> direction_scores;
  };

  static double sigmoid(double value)
  {
    if (value >= 0.0) {
      const double exp_value = std::exp(-value);
      return 1.0 / (1.0 + exp_value);
    }
    const double exp_value = std::exp(value);
    return exp_value / (1.0 + exp_value);
  }

  static double smoothstep01(double value)
  {
    const double t = std::clamp(value, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
  }

  static geometry_msgs::msg::Point to_point(const Eigen::Vector3d & value)
  {
    geometry_msgs::msg::Point point;
    point.x = value.x();
    point.y = value.y();
    point.z = value.z();
    return point;
  }

  static void append_vector3(std::vector<double> & values, const Eigen::Vector3d & vector)
  {
    values.push_back(vector.x());
    values.push_back(vector.y());
    values.push_back(vector.z());
  }

  static void append_joint_vector(std::vector<double> & values, const JointVector & vector)
  {
    values.insert(values.end(), vector.data(), vector.data() + vector.size());
  }

  static std_msgs::msg::ColorRGBA repulsion_metric_color(double metric_norm)
  {
    const double clamped_metric = std::clamp(metric_norm, 0.0, 1.0);
    std_msgs::msg::ColorRGBA color;
    color.r = static_cast<float>(std::min(1.0, 2.0 * clamped_metric));
    color.g = static_cast<float>(std::min(1.0, 2.0 * (1.0 - clamped_metric)));
    color.b = 0.05F;
    color.a = static_cast<float>(0.35 + 0.65 * clamped_metric);
    return color;
  }

  std::optional<RepulsionMetricSample> make_repulsion_metric_sample(
    const ControlPoint & control_point,
    const Eigen::Vector3d & control_point_velocity,
    const ObstacleSphere & obstacle) const
  {
    if (obstacle.radius <= 0.0) {
      return std::nullopt;
    }

    const Eigen::Vector3d delta = control_point.position - obstacle.center;
    const double center_distance = delta.norm();
    Eigen::Vector3d direction = Eigen::Vector3d::UnitZ();
    if (center_distance > 1e-9) {
      direction = delta / center_distance;
    }
    const double clearance = std::max(
      center_distance - (control_point.radius + obstacle.radius) -
      collision_metric_params_.margin,
      0.0);
    const double clearance_velocity = direction.dot(control_point_velocity);

    const double metric_radius =
      std::max(collision_metric_params_.metric_modulation_radius, 1e-9);
    const double metric_exploder_std_dev =
      std::max(collision_metric_params_.metric_exploder_std_dev, 1e-9);
    const double metric_exploder_eps =
      std::max(collision_metric_params_.metric_exploder_eps, 1e-9);
    const double metric_normalizer =
      std::max(std::abs(collision_metric_params_.metric_scalar) / metric_exploder_eps, 1e-9);

    double metric_gate =
      clearance * clearance / (metric_radius * metric_radius) -
      2.0 * clearance / metric_radius + 1.0;
    if (clearance > metric_radius) {
      metric_gate = 0.0;
    }

    const double distance_metric =
      collision_metric_params_.metric_scalar /
      (clearance / metric_exploder_std_dev + metric_exploder_eps) *
      metric_gate;
    const double velocity_gate_scale =
      std::max(collision_metric_params_.damping_velocity_gate_length_scale, 1e-9);
    const double sigma = sigmoid(clearance_velocity / velocity_gate_scale);
    const double effective_metric = clearance > metric_radius ?
      0.0 :
      distance_metric * (1.0 - sigma);

    const double repulsion_std_dev =
      std::max(collision_metric_params_.repulsion_std_dev, 1e-9);
    const double repulsion_norm =
      std::clamp(std::exp(-(clearance / repulsion_std_dev)), 0.0, 1.0);
    const double repel = collision_metric_params_.repulsion_gain * repulsion_norm;
    const double damping =
      -(1.0 - sigma) * collision_metric_params_.damping_gain * clearance_velocity /
      (clearance / std::max(collision_metric_params_.damping_std_dev, 1e-9) +
      std::max(collision_metric_params_.damping_robustness_eps, 1e-9));
    const double accel_normalizer =
      std::max(std::abs(collision_metric_params_.repulsion_gain), 1e-9);
    const double accel_norm =
      std::clamp(std::max(repel + damping, 0.0) / accel_normalizer, 0.0, 1.0);
    const double metric_norm =
      std::clamp(distance_metric / metric_normalizer, 0.0, 1.0);
    const double effective_metric_norm =
      std::clamp(effective_metric / metric_normalizer, 0.0, 1.0);
    const double intensity_norm =
      std::clamp(std::sqrt(effective_metric_norm * accel_norm), 0.0, 1.0);

    return RepulsionMetricSample{
      control_point.position,
      direction,
      clearance,
      metric_norm,
      effective_metric_norm,
      repulsion_norm,
      intensity_norm
    };
  }

  void publish_repulsion_metric_markers(
    const RobotState & state,
    const KinematicsContext & context)
  {
    if (!repulsion_metric_pub_) {
      return;
    }

    std::vector<std::optional<ObstacleSphere>> proximity_sensor_obstacles;
    proximity_sensor_obstacles_box_.get(proximity_sensor_obstacles);

    visualization_msgs::msg::MarkerArray markers;
    const auto stamp = now();
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "base_link";
    clear_marker.header.stamp = stamp;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear_marker);

    int marker_id = 0;
    for (std::size_t point_index = 0; point_index < context.control_points.size(); ++point_index) {
      if (
        point_index >= proximity_sensor_obstacles.size() ||
        !proximity_sensor_obstacles[point_index].has_value())
      {
        continue;
      }

      const auto & control_point = context.control_points[point_index];
      Eigen::Vector3d control_point_velocity = Eigen::Vector3d::Zero();
      if (point_index < context.control_point_jacobians.size()) {
        control_point_velocity = context.control_point_jacobians[point_index] * state.qd;
      }

      auto sample = make_repulsion_metric_sample(
        control_point,
        control_point_velocity,
        proximity_sensor_obstacles[point_index].value());
      if (
        !sample.has_value() ||
        sample->metric_norm < repulsion_metric_marker_min_norm_)
      {
        continue;
      }

      const double metric_norm = std::max(sample->metric_norm, sample->effective_metric_norm);

      visualization_msgs::msg::Marker sensor_marker;
      sensor_marker.header.frame_id = "base_link";
      sensor_marker.header.stamp = stamp;
      sensor_marker.ns = "repulsion_metric_dots";
      sensor_marker.id = marker_id++;
      sensor_marker.type = visualization_msgs::msg::Marker::SPHERE;
      sensor_marker.action = visualization_msgs::msg::Marker::ADD;
      sensor_marker.pose.position = to_point(sample->control_point);
      sensor_marker.pose.orientation.w = 1.0;
      sensor_marker.scale.x = repulsion_metric_marker_dot_diameter_;
      sensor_marker.scale.y = repulsion_metric_marker_dot_diameter_;
      sensor_marker.scale.z = repulsion_metric_marker_dot_diameter_;
      sensor_marker.color = repulsion_metric_color(metric_norm);
      markers.markers.push_back(sensor_marker);
    }

    repulsion_metric_pub_->publish(markers);
  }

  static Eigen::Vector3d compute_tcp_acceleration(
    const KinematicsContext & context,
    const JointVector & qdd)
  {
    return context.tcp_jacobian * qdd + context.tcp_curvature;
  }

  double tangent_escape_activation(double clearance) const
  {
    if (clearance >= tangent_escape_influence_distance_) {
      return 0.0;
    }
    if (clearance <= tangent_escape_safe_distance_) {
      return 1.0;
    }

    const double span =
      std::max(tangent_escape_influence_distance_ - tangent_escape_safe_distance_, 1e-9);
    const double normalized = (clearance - tangent_escape_safe_distance_) / span;
    return 1.0 - smoothstep01(normalized);
  }

  std::optional<Eigen::Vector3d> project_to_tangent_direction(
    const Eigen::Vector3d & direction,
    const Eigen::Vector3d & normal) const
  {
    if (!direction.allFinite()) {
      return std::nullopt;
    }

    Eigen::Vector3d tangent = direction - direction.dot(normal) * normal;
    const double tangent_norm = tangent.norm();
    if (tangent_norm <= tangent_escape_min_tangent_norm_) {
      return std::nullopt;
    }
    return tangent / tangent_norm;
  }

  void append_unique_escape_direction(
    std::vector<Eigen::Vector3d> & directions,
    const Eigen::Vector3d & candidate,
    const Eigen::Vector3d & normal) const
  {
    const auto tangent = project_to_tangent_direction(candidate, normal);
    if (!tangent.has_value()) {
      return;
    }

    for (const auto & direction : directions) {
      if (direction.dot(tangent.value()) > 0.98) {
        return;
      }
    }
    directions.push_back(tangent.value());
  }

  std::vector<Eigen::Vector3d> make_tangent_escape_directions(
    const Eigen::Vector3d & normal,
    const Eigen::Vector3d & goal_direction,
    const Eigen::Vector3d & point_velocity) const
  {
    std::vector<Eigen::Vector3d> directions;
    directions.reserve(static_cast<std::size_t>(tangent_escape_candidate_count_ + 10));

    append_unique_escape_direction(directions, goal_direction, normal);
    append_unique_escape_direction(directions, -goal_direction, normal);
    if (tangent_escape_previous_tangent_valid_) {
      append_unique_escape_direction(directions, tangent_escape_previous_tangent_, normal);
      append_unique_escape_direction(directions, -tangent_escape_previous_tangent_, normal);
    }
    append_unique_escape_direction(directions, point_velocity, normal);
    append_unique_escape_direction(directions, -point_velocity, normal);
    append_unique_escape_direction(directions, Eigen::Vector3d::UnitZ(), normal);
    append_unique_escape_direction(directions, -Eigen::Vector3d::UnitZ(), normal);
    append_unique_escape_direction(directions, Eigen::Vector3d::UnitX(), normal);
    append_unique_escape_direction(directions, -Eigen::Vector3d::UnitX(), normal);
    append_unique_escape_direction(directions, Eigen::Vector3d::UnitY(), normal);
    append_unique_escape_direction(directions, -Eigen::Vector3d::UnitY(), normal);

    const Eigen::Vector3d reference =
      std::abs(normal.z()) < 0.9 ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitY();
    Eigen::Vector3d basis_u = reference.cross(normal);
    if (basis_u.norm() <= 1e-9) {
      basis_u = Eigen::Vector3d::UnitX().cross(normal);
    }
    basis_u.normalize();
    const Eigen::Vector3d basis_v = normal.cross(basis_u).normalized();
    const int candidate_count = std::max(tangent_escape_candidate_count_, 4);
    for (int index = 0; index < candidate_count; ++index) {
      const double angle = 2.0 * kPi * static_cast<double>(index) /
        static_cast<double>(candidate_count);
      append_unique_escape_direction(
        directions,
        std::cos(angle) * basis_u + std::sin(angle) * basis_v,
        normal);
    }

    if (directions.empty()) {
      directions.push_back(basis_u);
    }
    return directions;
  }

  static std::string sensor_direction_suffix(const std::string & frame_name)
  {
    const auto separator = frame_name.find_last_of('_');
    if (separator == std::string::npos || separator + 1 >= frame_name.size()) {
      return frame_name;
    }
    return frame_name.substr(separator + 1);
  }

  std::optional<std::size_t> paired_sensor_index(std::size_t sensor_index) const
  {
    if (sensor_index >= RB10Model::sensor_control_points.size()) {
      return std::nullopt;
    }

    const auto & sensor = RB10Model::sensor_control_points[sensor_index];
    const std::string direction = sensor_direction_suffix(sensor.frame_name);
    std::optional<std::size_t> best_index;
    double best_distance = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < RB10Model::sensor_control_points.size(); ++index) {
      if (index == sensor_index) {
        continue;
      }
      const auto & candidate = RB10Model::sensor_control_points[index];
      if (candidate.parent_link != sensor.parent_link) {
        continue;
      }
      if (sensor_direction_suffix(candidate.frame_name) != direction) {
        continue;
      }
      const double distance = (candidate.offset - sensor.offset).norm();
      if (distance < 0.05 || distance >= best_distance) {
        continue;
      }
      best_index = index;
      best_distance = distance;
    }
    return best_index;
  }

  double proximity_sensor_detection(
    std::size_t sensor_index,
    const KinematicsContext & context,
    const std::vector<std::optional<ObstacleSphere>> & proximity_sensor_obstacles) const
  {
    if (
      sensor_index >= context.control_points.size() ||
      sensor_index >= proximity_sensor_obstacles.size() ||
      !proximity_sensor_obstacles[sensor_index].has_value())
    {
      return 0.0;
    }

    const auto & control_point = context.control_points[sensor_index];
    const auto & obstacle = proximity_sensor_obstacles[sensor_index].value();
    if (obstacle.radius <= 0.0 || !obstacle.center.allFinite()) {
      return 0.0;
    }

    const double clearance =
      (control_point.position - obstacle.center).norm() -
      (control_point.radius + obstacle.radius) -
      collision_metric_params_.margin;
    return tangent_escape_activation(clearance);
  }

  double score_tangent_escape_duplicate_risk(
    std::size_t active_sensor_index,
    const Eigen::Vector3d & direction,
    const Eigen::Vector3d & normal,
    const KinematicsContext & context,
    const std::vector<std::optional<ObstacleSphere>> & proximity_sensor_obstacles) const
  {
    const auto pair_index = paired_sensor_index(active_sensor_index);
    if (
      !pair_index.has_value() ||
      active_sensor_index >= context.control_points.size() ||
      pair_index.value() >= context.control_points.size())
    {
      return 0.0;
    }

    const auto pair_axis = project_to_tangent_direction(
      context.control_points[pair_index.value()].position -
      context.control_points[active_sensor_index].position,
      normal);
    if (!pair_axis.has_value()) {
      return 0.0;
    }

    const double paired_detection =
      proximity_sensor_detection(pair_index.value(), context, proximity_sensor_obstacles);
    const double toward_pair = std::max(0.0, direction.dot(pair_axis.value()));
    return paired_detection * toward_pair * toward_pair;
  }

  double score_tangent_escape_adjacent_block(
    std::size_t active_sensor_index,
    const Eigen::Vector3d & direction,
    const Eigen::Vector3d & normal,
    const KinematicsContext & context,
    const std::vector<std::optional<ObstacleSphere>> & proximity_sensor_obstacles) const
  {
    if (active_sensor_index >= RB10Model::sensor_control_points.size()) {
      return 0.0;
    }

    const auto pair_index = paired_sensor_index(active_sensor_index);
    const auto & active_sensor = RB10Model::sensor_control_points[active_sensor_index];
    double penalty = 0.0;
    for (std::size_t index = 0; index < RB10Model::sensor_control_points.size(); ++index) {
      if (
        index == active_sensor_index ||
        (pair_index.has_value() && index == pair_index.value()) ||
        index >= context.control_points.size())
      {
        continue;
      }
      const auto & sensor = RB10Model::sensor_control_points[index];
      if (sensor.parent_link != active_sensor.parent_link) {
        continue;
      }

      const double detection =
        proximity_sensor_detection(index, context, proximity_sensor_obstacles);
      if (detection <= 0.0) {
        continue;
      }

      const auto sensor_axis = project_to_tangent_direction(
        context.control_points[index].position -
        context.control_points[active_sensor_index].position,
        normal);
      if (!sensor_axis.has_value()) {
        continue;
      }
      const double alignment = std::max(0.0, direction.dot(sensor_axis.value()));
      penalty += detection * alignment * alignment;
    }

    return std::clamp(penalty, 0.0, 3.0);
  }

  TangentEscapeDirectionScore score_tangent_escape_direction(
    std::size_t direction_index,
    std::size_t active_sensor_index,
    const Eigen::Vector3d & direction,
    const Eigen::Vector3d & goal_direction,
    const Eigen::Vector3d & normal,
    const KinematicsContext & context,
    const std::vector<std::optional<ObstacleSphere>> & proximity_sensor_obstacles) const
  {
    const double goal_score = direction.dot(goal_direction);
    const double continuity_score = tangent_escape_previous_tangent_valid_ ?
      direction.dot(tangent_escape_previous_tangent_) :
      0.0;
    const double up_score = direction.z();
    const double duplicate_risk_score = score_tangent_escape_duplicate_risk(
      active_sensor_index,
      direction,
      normal,
      context,
      proximity_sensor_obstacles);
    const double adjacent_block_score = score_tangent_escape_adjacent_block(
      active_sensor_index,
      direction,
      normal,
      context,
      proximity_sensor_obstacles);
    const double branch_hold_score =
      tangent_escape_previous_tangent_valid_ &&
      tangent_escape_previous_control_point_index_ == active_sensor_index ?
      std::pow(std::max(0.0, direction.dot(tangent_escape_previous_tangent_)), 2.0) :
      0.0;

    TangentEscapeDirectionScore score;
    score.index = direction_index;
    score.direction = direction;
    score.goal_score = goal_score;
    score.continuity_score = continuity_score;
    score.up_score = up_score;
    score.duplicate_risk_score = duplicate_risk_score;
    score.adjacent_block_score = adjacent_block_score;
    score.branch_hold_score = branch_hold_score;
    score.total_score =
      tangent_escape_goal_weight_ * score.goal_score +
      tangent_escape_continuity_weight_ * score.continuity_score +
      tangent_escape_up_weight_ * score.up_score -
      tangent_escape_duplicate_risk_weight_ * score.duplicate_risk_score -
      tangent_escape_adjacent_block_weight_ * score.adjacent_block_score +
      tangent_escape_branch_hold_weight_ * score.branch_hold_score;
    return score;
  }

	  std::optional<TangentEscapeCandidate> select_tangent_escape_candidate(
	    const KinematicsContext & context,
	    const Eigen::Vector3d & goal,
	    const std::vector<ObstacleSphere> & obstacles,
    const std::vector<std::optional<ObstacleSphere>> & proximity_sensor_obstacles,
	    const JointVector & qd,
    const JointVector & qdd_raw) const
  {
    std::optional<TangentEscapeCandidate> best_candidate;
    double best_score = -std::numeric_limits<double>::infinity();

    for (std::size_t point_index = 0; point_index < context.control_points.size(); ++point_index) {
      if (point_index >= context.control_point_jacobians.size()) {
        continue;
      }

      const auto & control_point = context.control_points[point_index];
      const auto & jacobian = context.control_point_jacobians[point_index];
      const Eigen::Vector3d curvature =
        point_index < context.control_point_curvatures.size() ?
        context.control_point_curvatures[point_index] :
        Eigen::Vector3d::Zero();
      const Eigen::Vector3d raw_accel = jacobian * qdd_raw + curvature;

      for (const auto & obstacle : obstacles) {
        if (obstacle.radius <= 0.0 || !obstacle.center.allFinite()) {
          continue;
        }

        const Eigen::Vector3d delta = control_point.position - obstacle.center;
        const double center_distance = delta.norm();
        if (center_distance <= 1e-9) {
          continue;
        }

        const double clearance =
          center_distance - (control_point.radius + obstacle.radius) -
          collision_metric_params_.margin;
        const double activation = tangent_escape_activation(clearance);
        if (activation < tangent_escape_min_activation_) {
          continue;
        }

        const Eigen::Vector3d normal = delta / center_distance;
        const Eigen::Vector3d goal_delta = goal - control_point.position;
        const double goal_distance = goal_delta.norm();
        if (goal_distance <= 1e-9) {
          continue;
        }

        const Eigen::Vector3d goal_direction = goal_delta / goal_distance;
        const double goal_normal_dot = goal_direction.dot(normal);
        const Eigen::Vector3d point_velocity = jacobian * qd;
        const double inward_raw_accel = std::max(0.0, -normal.dot(raw_accel));
        const bool tangent_allowed =
          goal_normal_dot < tangent_escape_goal_normal_dot_threshold_;
        if (!tangent_allowed && inward_raw_accel <= 1e-9) {
          continue;
        }

        const auto escape_directions = make_tangent_escape_directions(
          normal,
          goal_direction,
          point_velocity);
        Eigen::Vector3d best_direction = escape_directions.front();
        std::size_t best_direction_index = 0;
        double best_direction_score = -std::numeric_limits<double>::infinity();
        std::vector<TangentEscapeDirectionScore> direction_scores;
        direction_scores.reserve(escape_directions.size());
        for (std::size_t direction_index = 0; direction_index < escape_directions.size();
          ++direction_index)
        {
          const auto & direction = escape_directions[direction_index];
          auto direction_score = score_tangent_escape_direction(
            direction_index,
            point_index,
            direction,
            goal_direction,
            normal,
            context,
            proximity_sensor_obstacles);
          if (direction_score.total_score > best_direction_score) {
            best_direction_score = direction_score.total_score;
            best_direction = direction;
            best_direction_index = direction_index;
          }
          direction_scores.push_back(direction_score);
        }
        if (best_direction_index < direction_scores.size()) {
          direction_scores[best_direction_index].selected = true;
        }

        const double inward_goal = std::max(0.0, -goal_normal_dot);
        const double score =
          activation *
          (1.0 + inward_goal + std::min(inward_raw_accel, 1.0) + best_direction_score);
        if (score <= best_score) {
          continue;
        }

        TangentEscapeCandidate candidate;
        candidate.control_point_index = point_index;
        candidate.control_point = control_point.position;
        candidate.obstacle_center = obstacle.center;
        candidate.normal = normal;
        candidate.tangent = best_direction;
        candidate.raw_accel = raw_accel;
        candidate.curvature = curvature;
        candidate.jacobian = jacobian;
        candidate.clearance = clearance;
        candidate.activation = activation;
        candidate.score = score;
        candidate.has_tangent = tangent_allowed;
        candidate.selected_direction_index = best_direction_index;
        candidate.direction_scores = std::move(direction_scores);
        best_candidate = candidate;
        best_score = score;
      }
    }

	    return best_candidate;
	  }

  TangentEscapeDebugSample make_tangent_escape_debug_sample(
    const TangentEscapeCandidate & candidate,
    const KinematicsContext & context,
    const JointVector & qdd_raw,
    const JointVector & qdd_filtered) const
  {
    TangentEscapeDebugSample sample;
    sample.control_point_index = candidate.control_point_index;
    sample.control_point = candidate.control_point;
    sample.obstacle_center = candidate.obstacle_center;
    sample.normal = candidate.normal;
    sample.tangent = candidate.tangent;
    sample.raw_accel = candidate.raw_accel;
    sample.filtered_accel = candidate.jacobian * qdd_filtered + candidate.curvature;
    sample.raw_tcp_accel = compute_tcp_acceleration(context, qdd_raw);
    sample.filtered_tcp_accel = compute_tcp_acceleration(context, qdd_filtered);
    sample.qdd_raw = qdd_raw;
    sample.qdd_filtered = qdd_filtered;
    sample.clearance = candidate.clearance;
    sample.activation = candidate.activation;
    sample.score = candidate.score;
    sample.has_tangent = candidate.has_tangent;
    return sample;
  }

  void publish_tangent_escape_filter_data(
    const std::optional<TangentEscapeDebugSample> & sample)
  {
    if (!tangent_escape_filter_data_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray msg;
    if (!sample.has_value()) {
      msg.data = {0.0};
    } else {
      msg.data.reserve(42);
      msg.data.push_back(1.0);
      msg.data.push_back(static_cast<double>(sample->control_point_index));
      msg.data.push_back(sample->clearance);
      msg.data.push_back(sample->activation);
      msg.data.push_back(sample->score);
      msg.data.push_back(sample->has_tangent ? 1.0 : 0.0);
      append_vector3(msg.data, sample->control_point);
      append_vector3(msg.data, sample->obstacle_center);
      append_vector3(msg.data, sample->normal);
      append_vector3(msg.data, sample->tangent);
      append_vector3(msg.data, sample->raw_accel);
      append_vector3(msg.data, sample->filtered_accel);
      append_vector3(msg.data, sample->raw_tcp_accel);
      append_vector3(msg.data, sample->filtered_tcp_accel);
      append_joint_vector(msg.data, sample->qdd_raw);
      append_joint_vector(msg.data, sample->qdd_filtered);
    }

    if (rt_tangent_escape_filter_data_pub_) {
      rt_tangent_escape_filter_data_pub_->tryPublish(msg);
    } else {
      tangent_escape_filter_data_pub_->publish(msg);
    }
  }

  void publish_tangent_escape_filter_candidate_data(
    const std::optional<TangentEscapeCandidate> & candidate)
  {
    if (!tangent_escape_filter_candidate_data_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray msg;
    if (!candidate.has_value()) {
      msg.data = {0.0};
    } else {
      msg.data.reserve(13 + candidate->direction_scores.size() * 18);
      msg.data.push_back(1.0);
      msg.data.push_back(static_cast<double>(candidate->direction_scores.size()));
      msg.data.push_back(static_cast<double>(candidate->selected_direction_index));
      msg.data.push_back(static_cast<double>(candidate->control_point_index));
      msg.data.push_back(candidate->clearance);
      msg.data.push_back(candidate->activation);
      msg.data.push_back(candidate->score);
      msg.data.push_back(tangent_escape_goal_weight_);
      msg.data.push_back(tangent_escape_continuity_weight_);
      msg.data.push_back(tangent_escape_up_weight_);
      msg.data.push_back(tangent_escape_duplicate_risk_weight_);
      msg.data.push_back(tangent_escape_adjacent_block_weight_);
      msg.data.push_back(tangent_escape_branch_hold_weight_);

      for (const auto & score : candidate->direction_scores) {
        msg.data.push_back(static_cast<double>(score.index));
        append_vector3(msg.data, score.direction);
        msg.data.push_back(score.goal_score);
        msg.data.push_back(score.continuity_score);
        msg.data.push_back(score.up_score);
        msg.data.push_back(score.duplicate_risk_score);
        msg.data.push_back(score.adjacent_block_score);
        msg.data.push_back(score.branch_hold_score);
        msg.data.push_back(tangent_escape_goal_weight_ * score.goal_score);
        msg.data.push_back(tangent_escape_continuity_weight_ * score.continuity_score);
        msg.data.push_back(tangent_escape_up_weight_ * score.up_score);
        msg.data.push_back(-tangent_escape_duplicate_risk_weight_ * score.duplicate_risk_score);
        msg.data.push_back(-tangent_escape_adjacent_block_weight_ * score.adjacent_block_score);
        msg.data.push_back(tangent_escape_branch_hold_weight_ * score.branch_hold_score);
        msg.data.push_back(score.total_score);
        msg.data.push_back(score.selected ? 1.0 : 0.0);
      }
    }

    if (rt_tangent_escape_filter_candidate_data_pub_) {
      rt_tangent_escape_filter_candidate_data_pub_->tryPublish(msg);
    } else {
      tangent_escape_filter_candidate_data_pub_->publish(msg);
    }
  }

  void publish_tangent_escape_debug_sample(
    const std::optional<TangentEscapeDebugSample> & sample)
  {
    if (publish_tangent_escape_filter_debug_enabled_) {
      TangentEscapeDebugState debug_state;
      if (sample.has_value()) {
        debug_state.samples.push_back(sample.value());
      }
      tangent_escape_debug_box_.set(debug_state);
    }
    publish_tangent_escape_filter_data(sample);
  }

	  JointVector apply_tangent_escape_filter(
	    const KinematicsContext & context,
	    const Eigen::Vector3d & goal,
    const std::vector<ObstacleSphere> & obstacles,
    const std::vector<std::optional<ObstacleSphere>> & proximity_sensor_obstacles,
    const JointVector & qd,
	    const JointVector & qdd_raw)
	  {
	    if (!enable_tangent_escape_filter_ || tangent_escape_max_delta_qdd_ <= 0.0) {
      publish_tangent_escape_filter_candidate_data(std::nullopt);
      publish_tangent_escape_debug_sample(std::nullopt);
	      return qdd_raw;
	    }

	    const auto candidate =
	      select_tangent_escape_candidate(
          context,
          goal,
          obstacles,
          proximity_sensor_obstacles,
          qd,
          qdd_raw);
	    if (!candidate.has_value()) {
      publish_tangent_escape_filter_candidate_data(std::nullopt);
      publish_tangent_escape_debug_sample(std::nullopt);
	      return qdd_raw;
	    }
    publish_tangent_escape_filter_candidate_data(candidate);

    Eigen::Vector3d desired_task_accel = candidate->raw_accel;
    if (tangent_escape_normal_gain_ > 0.0) {
      const double inward_raw_accel =
        std::min(0.0, candidate->normal.dot(desired_task_accel));
      desired_task_accel +=
        candidate->activation * tangent_escape_normal_gain_ * (-inward_raw_accel) *
        candidate->normal;
    }
    if (candidate->has_tangent) {
      const double desired_tangent_accel =
        candidate->activation * tangent_escape_tangent_gain_;
      const double current_tangent_accel = desired_task_accel.dot(candidate->tangent);
      if (current_tangent_accel < desired_tangent_accel) {
        desired_task_accel +=
          (desired_tangent_accel - current_tangent_accel) * candidate->tangent;
      }
    }

	    const Eigen::Vector3d delta_task_accel = desired_task_accel - candidate->raw_accel;

	    if (delta_task_accel.norm() <= 1e-9) {
      const auto sample = make_tangent_escape_debug_sample(
        candidate.value(),
        context,
        qdd_raw,
        qdd_raw);
      publish_tangent_escape_debug_sample(sample);
	      return qdd_raw;
	    }

	    Eigen::Matrix3d task_matrix = candidate->jacobian * candidate->jacobian.transpose();
	    task_matrix.diagonal().array() += tangent_escape_damping_ * tangent_escape_damping_;
	    const Eigen::Vector3d task_solution = task_matrix.ldlt().solve(delta_task_accel);
	    if (!task_solution.allFinite()) {
      publish_tangent_escape_debug_sample(std::nullopt);
	      return qdd_raw;
	    }

	    JointVector delta_qdd = candidate->jacobian.transpose() * task_solution;
	    if (!delta_qdd.allFinite()) {
      publish_tangent_escape_debug_sample(std::nullopt);
	      return qdd_raw;
	    }

    const double delta_norm = delta_qdd.norm();
    if (delta_norm > tangent_escape_max_delta_qdd_) {
      delta_qdd *= tangent_escape_max_delta_qdd_ / delta_norm;
    }

    JointVector qdd_filtered = qdd_raw + delta_qdd;
    for (int index = 0; index < qdd_filtered.size(); ++index) {
      qdd_filtered[index] = std::clamp(qdd_filtered[index], -max_joint_accel_, max_joint_accel_);
    }

    const auto sample = make_tangent_escape_debug_sample(
      candidate.value(),
      context,
      qdd_raw,
      qdd_filtered);
    publish_tangent_escape_debug_sample(sample);

	    tangent_escape_previous_tangent_ = candidate->tangent;
    tangent_escape_previous_control_point_index_ = candidate->control_point_index;
    tangent_escape_previous_tangent_valid_ = candidate->has_tangent;

    return qdd_filtered;
  }

  void update_tcp_accel_visualization(const Eigen::Vector3d & tcp_accel)
  {
    TcpAccelerationSample sample;
    sample.acceleration = {tcp_accel.x(), tcp_accel.y(), tcp_accel.z()};
    sample.norm = tcp_accel.norm();
    sample.valid = true;
    tcp_accel_visualization_box_.set(sample);
  }

  void publish_rmp_accel_debug(
    const JointVector & qdd,
    const Eigen::Vector3d * tcp_accel)
  {
    if (rmp_joint_accel_pub_) {
      std_msgs::msg::Float64MultiArray msg;
      msg.data.assign(qdd.data(), qdd.data() + qdd.size());
      if (rt_rmp_joint_accel_pub_) {
        rt_rmp_joint_accel_pub_->tryPublish(msg);
      } else {
        rmp_joint_accel_pub_->publish(msg);
      }
    }

    if (rmp_tcp_accel_pub_ && tcp_accel) {
      std_msgs::msg::Float64MultiArray msg;
      msg.data = {tcp_accel->x(), tcp_accel->y(), tcp_accel->z()};
      if (rt_rmp_tcp_accel_pub_) {
        rt_rmp_tcp_accel_pub_->tryPublish(msg);
      } else {
        rmp_tcp_accel_pub_->publish(msg);
      }
    }
  }

  void clear_tcp_accel_visualization_sample()
  {
    tcp_accel_visualization_box_.set(TcpAccelerationSample{});
  }

  void publish_tcp_accel_marker(const KinematicsContext & context)
  {
    if (!tcp_accel_pub_) {
      return;
    }

    TcpAccelerationSample sample;
    tcp_accel_visualization_box_.get(sample);

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "base_link";
    marker.header.stamp = now();
    marker.ns = "tcp_accel";
    marker.id = 0;

    if (!sample.valid || sample.norm < tcp_accel_marker_min_norm_) {
      marker.action = visualization_msgs::msg::Marker::DELETE;
      tcp_accel_pub_->publish(marker);
      return;
    }

    Eigen::Vector3d acceleration(
      sample.acceleration[0],
      sample.acceleration[1],
      sample.acceleration[2]);
    const double acceleration_norm = acceleration.norm();
    if (acceleration_norm < std::numeric_limits<double>::epsilon()) {
      marker.action = visualization_msgs::msg::Marker::DELETE;
      tcp_accel_pub_->publish(marker);
      return;
    }

    const double normalized_length =
      std::clamp(acceleration_norm / tcp_accel_marker_norm_for_max_length_, 0.0, 1.0);
    const double visible_length =
      tcp_accel_marker_max_length_ * std::max(normalized_length, 0.15);
    const Eigen::Vector3d start = context.tcp_position;
    const Eigen::Vector3d end = start + acceleration.normalized() * visible_length;

    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.points.push_back(to_point(start));
    marker.points.push_back(to_point(end));
    marker.scale.x = 0.01;
    marker.scale.y = 0.03;
    marker.scale.z = std::clamp(visible_length * 0.35, 0.015, 0.045);
    marker.color.r = 0.05F;
    marker.color.g = 0.85F;
    marker.color.b = 1.0F;
    marker.color.a = 0.9F;
    tcp_accel_pub_->publish(marker);
  }

  void publish_tangent_escape_filter_markers()
  {
    if (!tangent_escape_debug_pub_) {
      return;
    }

    TangentEscapeDebugState debug_state;
    tangent_escape_debug_box_.get(debug_state);

    visualization_msgs::msg::MarkerArray markers;
    const auto stamp = now();
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "base_link";
    clear_marker.header.stamp = stamp;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear_marker);

    auto color = [](float r, float g, float b, float a) {
        std_msgs::msg::ColorRGBA value;
        value.r = r;
        value.g = g;
        value.b = b;
        value.a = a;
        return value;
      };

    int marker_id = 0;
    auto append_sphere =
      [&](const Eigen::Vector3d & position, double diameter, const std_msgs::msg::ColorRGBA & rgba)
      {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = stamp;
        marker.ns = "tangent_escape_filter_points";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position = to_point(position);
        marker.pose.orientation.w = 1.0;
        marker.scale.x = diameter;
        marker.scale.y = diameter;
        marker.scale.z = diameter;
        marker.color = rgba;
        markers.markers.push_back(marker);
      };

    auto append_arrow =
      [&](const std::string & name,
        const Eigen::Vector3d & start,
        const Eigen::Vector3d & direction,
        double length,
        const std_msgs::msg::ColorRGBA & rgba)
      {
        const double direction_norm = direction.norm();
        if (direction_norm <= 1e-9) {
          return;
        }

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = stamp;
        marker.ns = name;
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::ARROW;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.points.push_back(to_point(start));
        marker.points.push_back(to_point(start + direction / direction_norm * length));
        marker.scale.x = 0.008;
        marker.scale.y = 0.024;
        marker.scale.z = 0.032;
        marker.color = rgba;
        markers.markers.push_back(marker);
      };

    for (const auto & sample : debug_state.samples) {
      const double length_scale =
        tangent_escape_marker_length_ * std::clamp(sample.activation, 0.25, 1.0);
      append_sphere(sample.control_point, 0.025, color(1.0F, 1.0F, 1.0F, 0.9F));
      append_sphere(sample.obstacle_center, 0.025, color(1.0F, 0.55F, 0.0F, 0.8F));
      append_arrow(
        "tangent_escape_filter_normal",
        sample.control_point,
        sample.normal,
        length_scale,
        color(1.0F, 0.05F, 0.05F, 0.9F));
      if (sample.has_tangent) {
        append_arrow(
          "tangent_escape_filter_tangent",
          sample.control_point,
          sample.tangent,
          length_scale,
          color(0.0F, 0.9F, 0.25F, 0.9F));
      }
      append_arrow(
        "tangent_escape_filter_raw_accel",
        sample.control_point,
        sample.raw_accel,
        tangent_escape_marker_length_,
        color(0.0F, 0.25F, 1.0F, 0.75F));
      append_arrow(
        "tangent_escape_filter_filtered_accel",
        sample.control_point,
        sample.filtered_accel,
        tangent_escape_marker_length_,
        color(0.0F, 0.9F, 1.0F, 0.85F));
    }

    tangent_escape_debug_pub_->publish(markers);
  }

  std::array<double, 6> parse_joint_limit_degrees(
    const std::string & parameter_name,
    const std::array<double, 6> & fallback) const
  {
    const auto values = get_parameter(parameter_name).as_double_array();
    if (values.empty()) {
      return fallback;
    }
    if (values.size() != fallback.size()) {
      throw std::runtime_error(parameter_name + " must contain exactly 6 values");
    }

    std::array<double, 6> limits{};
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (!std::isfinite(values[index])) {
        throw std::runtime_error(parameter_name + " must contain finite values");
      }
      limits[index] = degrees_to_radians(values[index]);
    }
    return limits;
  }

  JointVector clamp_joint_positions(const JointVector & q) const
  {
    JointVector clamped = q;
    for (int index = 0; index < clamped.size(); ++index) {
      const auto limit_index = static_cast<std::size_t>(index);
      clamped[index] = std::clamp(
        clamped[index],
        joint_lower_limits_[limit_index],
        joint_upper_limits_[limit_index]);
    }
    return clamped;
  }

  JointVector compute_acceleration(
    const RobotState & state,
    const Eigen::Vector3d & goal,
    const Eigen::Quaterniond & goal_orientation,
    const std::vector<ObstacleSphere> & obstacles) const
  {
    std::unordered_map<std::string, Eigen::Vector3d> vector_targets;
    vector_targets.emplace("goal", goal);
    vector_targets.emplace("body_goal", body_goal_);
    const Eigen::Matrix3d goal_rotation = goal_orientation.normalized().toRotationMatrix();
    vector_targets.emplace("orientation_goal_x", goal_rotation.col(0));
    vector_targets.emplace("orientation_goal_y", goal_rotation.col(1));
    vector_targets.emplace("orientation_goal_z", goal_rotation.col(2));
    std::unordered_map<std::string, ExternalRmpFeature> external_rmps;
    {
      std::scoped_lock lock(external_rmp_mutex_);
      for (const auto & entry : external_rmp_buffers_) {
        const auto & buffer = entry.second;
        if (!buffer.has_metric || !buffer.has_acceleration) {
          continue;
        }
        external_rmps.emplace(
          entry.first,
          ExternalRmpFeature{buffer.metric_sqrt, buffer.acceleration});
      }
    }
    const auto solution = solver_->solve(
      state.q,
      state.qd,
      vector_targets,
      obstacles,
      external_rmps);
    JointVector qdd = solution.qdd;
    for (int index = 0; index < qdd.size(); ++index) {
      qdd[index] = std::clamp(qdd[index], -max_joint_accel_, max_joint_accel_);
    }
    return qdd;
  }

  RobotState integrate_command(
    const RobotState & state,
    const JointVector & qdd,
    double dt) const
  {
    RobotState command_state = state;
    command_state.qd += qdd * dt;
    command_state.q += command_state.qd * dt;
    command_state.q = clamp_joint_positions(command_state.q);

    for (int index = 0; index < command_state.q.size(); ++index) {
      const auto limit_index = static_cast<std::size_t>(index);
      if (
        (command_state.q[index] <= joint_lower_limits_[limit_index] + 0.1 &&
        command_state.qd[index] < 0.0) ||
        (command_state.q[index] >= joint_upper_limits_[limit_index] - 0.1 &&
        command_state.qd[index] > 0.0))
      {
        command_state.qd[index] *= 0.5;
      }
    }

    return command_state;
  }

  void send_velocity_hold_command(const RobotState & measured_state)
  {
    if (command_mode_ != CommandMode::kVelocity || !backend_) {
      return;
    }

    RobotState hold_state = measured_state;
    hold_state.qd.setZero();
    backend_->apply_command(hold_state, RB10Model::joint_names);
    publish_position_command(hold_state);
    publish_direct_command(hold_state);
    virtual_velocity_state_ = hold_state;
    virtual_velocity_state_initialized_ = true;
  }

  void reset_controller_velocity_estimator()
  {
    controller_velocity_estimator_initialized_ = false;
    controller_velocity_previous_q_.setZero();
    controller_velocity_estimate_.setZero();
  }

  JointVector estimate_controller_velocity(
    const JointVector & q,
    double dt)
  {
    if (!controller_velocity_estimator_initialized_ || dt <= std::numeric_limits<double>::epsilon()) {
      controller_velocity_previous_q_ = q;
      controller_velocity_estimate_.setZero();
      controller_velocity_estimator_initialized_ = true;
      return controller_velocity_estimate_;
    }

    const JointVector raw_qd = (q - controller_velocity_previous_q_) / dt;
    controller_velocity_previous_q_ = q;
    controller_velocity_estimate_ =
      controller_velocity_filter_alpha_ * raw_qd +
      (1.0 - controller_velocity_filter_alpha_) * controller_velocity_estimate_;
    return controller_velocity_estimate_;
  }

  void publish_rmp_ee_pose(const KinematicsContext & context)
  {
    if (!rmp_eef_pose_pub_) {
      return;
    }

    geometry_msgs::msg::Pose pose;
    pose.position.x = context.tcp_position.x();
    pose.position.y = context.tcp_position.y();
    pose.position.z = context.tcp_position.z();

    Eigen::Quaterniond tcp_orientation(context.link_rotations[RB10Model::TCP_RMP]);
    tcp_orientation.normalize();
    pose.orientation.x = tcp_orientation.x();
    pose.orientation.y = tcp_orientation.y();
    pose.orientation.z = tcp_orientation.z();
    pose.orientation.w = tcp_orientation.w();

    if (rt_rmp_eef_pose_pub_) {
      rt_rmp_eef_pose_pub_->tryPublish(pose);
      return;
    }
    rmp_eef_pose_pub_->publish(pose);
  }

  void publish_target_metric(const RobotState & state, const Eigen::Vector3d & goal)
  {
    if (!target_metric_pub_) {
      return;
    }

    const auto context = RB10Model::forward_context(state.q);
    const Eigen::Vector3d delta = goal - context.tcp_position;
    const double delta_norm = delta.norm();
    const double soft_delta_norm =
      std::max(delta_norm, target_metric_params_.accel_norm_eps / 10.0);
    const Eigen::Vector3d delta_hat = delta / soft_delta_norm;

    const Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d shape = delta_hat * delta_hat.transpose();
    const double scaled_dist = delta_norm / target_metric_params_.metric_alpha_length_scale;
    const double alpha =
      (1.0 - target_metric_params_.min_metric_alpha) *
      std::exp(-0.5 * scaled_dist * scaled_dist) +
      target_metric_params_.min_metric_alpha;
    Eigen::Matrix3d leaf_metric =
      alpha * target_metric_params_.max_metric_scalar * eye +
      (1.0 - alpha) * target_metric_params_.min_metric_scalar * shape;

    const double boost_scaled_dist =
      delta_norm / target_metric_params_.proximity_metric_boost_length_scale;
    const double boost_alpha = std::exp(-0.5 * boost_scaled_dist * boost_scaled_dist);
    const double metric_boost_scalar =
      boost_alpha * target_metric_params_.proximity_metric_boost_scalar +
      (1.0 - boost_alpha);
    leaf_metric *= metric_boost_scalar;

    std_msgs::msg::Float64MultiArray metric_msg;
    metric_msg.data.reserve(9);
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        metric_msg.data.push_back(leaf_metric(row, col));
      }
    }

    if (rt_target_metric_pub_) {
      rt_target_metric_pub_->tryPublish(metric_msg);
      return;
    }
    target_metric_pub_->publish(metric_msg);
  }

  void publish_goal_tf(const GoalTarget & goal_target)
  {
    if (!goal_tf_broadcaster_ || goal_tf_parent_frame_.empty() || goal_tf_child_frame_.empty()) {
      return;
    }

    Eigen::Quaterniond goal_orientation = goal_target.orientation;
    if (!goal_orientation.coeffs().allFinite() || goal_orientation.norm() < 1e-9) {
      goal_orientation = Eigen::Quaterniond::Identity();
    } else {
      goal_orientation.normalize();
    }

    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = now();
    transform.header.frame_id = goal_tf_parent_frame_;
    transform.child_frame_id = goal_tf_child_frame_;
    transform.transform.translation.x = goal_target.position.x();
    transform.transform.translation.y = goal_target.position.y();
    transform.transform.translation.z = goal_target.position.z();
    transform.transform.rotation.x = goal_orientation.x();
    transform.transform.rotation.y = goal_orientation.y();
    transform.transform.rotation.z = goal_orientation.z();
    transform.transform.rotation.w = goal_orientation.w();
    goal_tf_broadcaster_->sendTransform(transform);
  }

  bool violates_min_z_safety(const KinematicsContext & context) const
  {
    if (rmp_flag_gate_enabled_ && !rmp_active_.load()) {
      return false;
    }

    if (!safety_stop_on_min_z_) {
      return false;
    }

    const auto floor_metrics = compute_floor_safety_metrics(context);
    return
      floor_metrics.min_link_z < workspace_floor_z_ ||
      floor_metrics.min_joint_z < workspace_floor_z_ ||
      floor_metrics.min_control_point_z < workspace_floor_z_ ||
      floor_metrics.min_body_obstacle_z < workspace_floor_z_ ||
      context.link_positions[RB10Model::LINK6].z() < min_link6_z_ ||
      context.link_positions[RB10Model::TCP_GRIPPER].z() < min_tcp_z_;
  }

  bool apply_command_guard(
    const RobotState & measured_state,
    RobotState & command_state,
    double dt) const
  {
    bool clamped = false;
    if (command_mode_ == CommandMode::kVelocity) {
      const double safe_dt = std::max(dt, 1e-6);
      for (int index = 0; index < command_state.q.size(); ++index) {
        const auto limit_index = static_cast<std::size_t>(index);
        double limited_velocity = std::clamp(
          command_state.qd[index],
          -command_guard_max_velocity_rad_s_,
          command_guard_max_velocity_rad_s_);
        double guarded_lower = joint_lower_limits_[limit_index];
        double guarded_upper = joint_upper_limits_[limit_index];

        if (predictive_joint_limit_guard_) {
          const double raw_lower = joint_lower_limits_[limit_index];
          const double raw_upper = joint_upper_limits_[limit_index];
          const double max_buffer = std::max(0.0, 0.45 * (raw_upper - raw_lower));
          const double buffer = std::min(joint_limit_buffers_[limit_index], max_buffer);
          guarded_lower = raw_lower + buffer;
          guarded_upper = raw_upper - buffer;
          const double measured_q = measured_state.q[index];

          if (measured_q < guarded_lower) {
            limited_velocity = std::max(limited_velocity, 0.0);
          } else if (measured_q > guarded_upper) {
            limited_velocity = std::min(limited_velocity, 0.0);
          } else {
            const double lower_velocity = (guarded_lower - measured_q) / safe_dt;
            const double upper_velocity = (guarded_upper - measured_q) / safe_dt;
            limited_velocity = std::clamp(limited_velocity, lower_velocity, upper_velocity);
          }
        } else if (
          (measured_state.q[index] <= joint_lower_limits_[limit_index] + 0.1 &&
          limited_velocity < 0.0) ||
          (measured_state.q[index] >= joint_upper_limits_[limit_index] - 0.1 &&
          limited_velocity > 0.0))
        {
          limited_velocity = 0.0;
        }
        if (std::abs(limited_velocity - command_state.qd[index]) > 1e-9) {
          clamped = true;
        }
        command_state.qd[index] = limited_velocity;
        command_state.q[index] = measured_state.q[index] + command_state.qd[index] * dt;
        if (predictive_joint_limit_guard_) {
          command_state.q[index] = std::clamp(command_state.q[index], guarded_lower, guarded_upper);
        }
      }
      command_state.q = clamp_joint_positions(command_state.q);
      return clamped;
    }

    const double max_step_rad =
      std::max(command_guard_max_step_rad_, command_guard_max_velocity_rad_s_ * dt);

    for (int index = 0; index < command_state.q.size(); ++index) {
      const double delta = command_state.q[index] - measured_state.q[index];
      const double limited_delta = std::clamp(delta, -max_step_rad, max_step_rad);
      if (std::abs(limited_delta - delta) > 1e-9) {
        clamped = true;
      }
      command_state.q[index] = measured_state.q[index] + limited_delta;

      const double limited_velocity = std::clamp(
        command_state.qd[index],
        -command_guard_max_velocity_rad_s_,
        command_guard_max_velocity_rad_s_);
      if (std::abs(limited_velocity - command_state.qd[index]) > 1e-9) {
        clamped = true;
      }
      command_state.qd[index] = limited_velocity;
    }

    const JointVector unclamped_q = command_state.q;
    command_state.q = clamp_joint_positions(command_state.q);
    if ((command_state.q - unclamped_q).cwiseAbs().maxCoeff() > 1e-9) {
      clamped = true;
    }

    return clamped;
  }

  void control_loop()
  {
    enable_realtime(
      get_logger(),
      enable_realtime_,
      realtime_priority_,
      lock_memory_);

    const auto period = std::chrono::duration<double>(1.0 / control_rate_hz_);
    auto next_tick = std::chrono::steady_clock::now();

    while (rclcpp::ok() && running_.load()) {
      next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);

      if (!backend_->ready()) {
        virtual_velocity_state_initialized_ = false;
        last_safe_command_state_initialized_ = false;
        clear_tcp_accel_visualization_sample();
        reset_controller_velocity_estimator();
        RCLCPP_INFO_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Waiting for initial robot state on %s before enabling RMP commands",
          joint_state_topic_.c_str());
        std::this_thread::sleep_until(next_tick);
        continue;
      }

      RobotState measured_state = backend_->read_state();
      if (estimate_velocity_in_controller_) {
        // Estimate qd on the controller's own 100 Hz timeline so the solver
        // sees a velocity signal aligned with the control loop instead of a
        // faster, asynchronously sampled derivative from the bridge.
        measured_state.qd = estimate_controller_velocity(measured_state.q, period.count());
      }
      const auto measured_context = RB10Model::forward_context(measured_state.q);
      if (rmp_flag_gate_enabled_ && !rmp_active_.load()) {
        virtual_velocity_state_initialized_ = false;
        last_safe_command_state_initialized_ = false;
        clear_tcp_accel_visualization_sample();
        state_ = measured_state;
        if (estimate_velocity_in_controller_) {
          state_.qd = measured_state.qd;
        }
        if (publish_joint_states_enabled_) {
          publish_joint_states(state_);
        }
        publish_rmp_ee_pose(measured_context);
        GoalTarget goal_target;
        goal_target_box_.get(goal_target);
        publish_target_metric(measured_state, goal_target.position);
        send_velocity_hold_command(measured_state);
        RCLCPP_INFO_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "RMP standby: waiting for /RMP_flag == %d before running the solve loop",
          rmp_active_flag_value_);
        std::this_thread::sleep_until(next_tick);
        continue;
      }
      if (rmp_flag_gate_enabled_ && !external_goal_received_.load()) {
        virtual_velocity_state_initialized_ = false;
        last_safe_command_state_initialized_ = false;
        clear_tcp_accel_visualization_sample();
        state_ = measured_state;
        if (estimate_velocity_in_controller_) {
          state_.qd = measured_state.qd;
        }
        if (publish_joint_states_enabled_) {
          publish_joint_states(state_);
        }
        publish_rmp_ee_pose(measured_context);
        GoalTarget goal_target;
        goal_target_box_.get(goal_target);
        publish_target_metric(measured_state, goal_target.position);
        send_velocity_hold_command(measured_state);
        RCLCPP_INFO_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "RMP active but waiting for /goal_pose before running the solve loop");
        std::this_thread::sleep_until(next_tick);
        continue;
      }
      if (
        initialize_goal_from_first_state_ &&
        !startup_goal_synced_.load() &&
        !external_goal_received_.load())
      {
        const Eigen::Vector3d startup_goal =
          measured_context.link_positions[RB10Model::TCP_RMP];
        Eigen::Quaterniond startup_orientation(
          measured_context.link_rotations[RB10Model::TCP_RMP]);
        startup_orientation.normalize();
        goal_target_box_.set(GoalTarget{startup_goal, startup_orientation});
        startup_goal_synced_.store(true);
        RCLCPP_INFO(
          get_logger(),
          "Initialized startup goal from the current tcp_rmp pose to avoid a jump at launch");
      }
      const bool measured_state_violates_min_z = violates_min_z_safety(measured_context);
      if (!last_safe_command_state_initialized_ && !measured_state_violates_min_z) {
        last_safe_command_state_ = measured_state;
        last_safe_command_state_.qd.setZero();
        last_safe_command_state_initialized_ = true;
      }

      RobotState control_state = measured_state;
      if (virtual_velocity_state_initialized_) {
        // Blending measured q with the previous commanded q lets the outer
        // loop stay responsive when the robot's internal servo is still
        // catching up, while preserving an easy path back to pure measured
        // feedback for safety-oriented testing.
        control_state.q =
          measured_position_feedback_blend_ * measured_state.q +
          (1.0 - measured_position_feedback_blend_) * virtual_velocity_state_.q;

        // Blend measured and commanded joint velocity. Pure commanded velocity
        // is responsive but can hunt on real hardware, while pure measured
        // velocity tends to become too sluggish because it reflects the robot's
        // lagging servo response and filtered state estimate.
        control_state.qd =
          measured_velocity_feedback_blend_ * measured_state.qd +
          (1.0 - measured_velocity_feedback_blend_) * virtual_velocity_state_.qd;
      }
      if (!use_velocity_feedback_in_solver_) {
        // Allow testing the target attractor with position-only feedback by
        // removing the joint velocity term from the solver input.
        control_state.qd.setZero();
      }
      Eigen::Vector3d goal;
      Eigen::Quaterniond goal_orientation;
      std::vector<ObstacleSphere> obstacles;
      std::vector<std::optional<ObstacleSphere>> proximity_sensor_obstacles;
      GoalTarget goal_target;
      goal_target_box_.get(goal_target);
      goal = goal_target.position;
      goal_orientation = goal_target.orientation;
      obstacles_box_.get(obstacles);
      proximity_sensor_obstacles_box_.get(proximity_sensor_obstacles);

      JointVector qdd = JointVector::Zero();
      last_min_z_safety_triggered_.store(false);
      bool solve_ok = false;
      try {
        qdd = compute_acceleration(
          control_state,
          goal,
          goal_orientation,
          obstacles);
        solve_ok = true;
      } catch (const std::exception & error) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "RMP solve failed, holding command at zero acceleration: %s",
          error.what());
      }

      std::optional<KinematicsContext> control_context_storage;
      auto control_context = [&]() -> const KinematicsContext & {
          if ((control_state.q - measured_state.q).cwiseAbs().maxCoeff() <= 1e-12) {
            return measured_context;
          }
          if (!control_context_storage.has_value()) {
            control_context_storage.emplace(RB10Model::forward_context(control_state.q));
          }
          return control_context_storage.value();
        };

      if (solve_ok && enable_tangent_escape_filter_) {
        qdd = apply_tangent_escape_filter(
          control_context(),
          goal,
          obstacles,
          proximity_sensor_obstacles,
          control_state.qd,
          qdd);
      }

      std::optional<Eigen::Vector3d> tcp_accel;
      if (tcp_accel_pub_ || rmp_tcp_accel_pub_) {
        tcp_accel = compute_tcp_acceleration(control_context(), qdd);
      }
      const Eigen::Vector3d * tcp_accel_ptr = tcp_accel ? &(*tcp_accel) : nullptr;
      if (tcp_accel_pub_ && tcp_accel_ptr) {
        update_tcp_accel_visualization(*tcp_accel_ptr);
      }
      publish_rmp_accel_debug(qdd, tcp_accel_ptr);
      RobotState predicted_state = integrate_command(control_state, qdd, period.count());
      const auto predicted_context = RB10Model::forward_context(predicted_state.q);
      RobotState command_state = predicted_state;
      const bool predicted_state_violates_min_z = violates_min_z_safety(predicted_context);
      const KinematicsContext * command_context = &predicted_context;
      std::optional<KinematicsContext> safe_command_context;
      const bool previous_min_z_safety_triggered = last_min_z_safety_triggered_.load();
      const bool min_z_safety_triggered =
        measured_state_violates_min_z || predicted_state_violates_min_z;
      if (min_z_safety_triggered) {
        if (last_safe_command_state_initialized_) {
          command_state = last_safe_command_state_;
          safe_command_context.emplace(RB10Model::forward_context(command_state.q));
          command_context = &safe_command_context.value();
        } else {
          command_state = measured_state;
          command_context = &measured_context;
        }
        command_state.qd.setZero();
        last_min_z_safety_triggered_.store(true);
        if (!previous_min_z_safety_triggered) {
          RCLCPP_ERROR(
            get_logger(),
            "Min-Z safety triggered. Holding the last safe command to avoid driving below workspace.");
        }
      }

      const bool command_guard_triggered =
        apply_command_guard(measured_state, command_state, period.count());
      if (command_guard_triggered) {
        safe_command_context.emplace(RB10Model::forward_context(command_state.q));
        command_context = &safe_command_context.value();
        RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          1000,
          command_mode_ == CommandMode::kVelocity ?
          "Command guard limited the joint velocity command to avoid triggering robot safety." :
          "Command guard limited the per-cycle joint step/velocity to avoid triggering robot safety.");
      }

      if (!min_z_safety_triggered) {
        if (previous_min_z_safety_triggered) {
          RCLCPP_INFO(get_logger(), "Min-Z safety cleared.");
        }
        last_safe_command_state_ = command_state;
        last_safe_command_state_initialized_ = true;
      }
      publish_rmp_ee_pose(*command_context);
      publish_target_metric(control_state, goal);
      publish_target_q(command_state);
      publish_position_command(command_state);
      backend_->apply_command(command_state, RB10Model::joint_names);
      publish_direct_command(command_state);
      virtual_velocity_state_ = command_state;
      virtual_velocity_state_initialized_ = true;
      state_ = backend_->read_state();
      if (estimate_velocity_in_controller_) {
        state_.qd = measured_state.qd;
      }
      if (publish_joint_states_enabled_) {
        publish_joint_states(state_);
      }

      std::this_thread::sleep_until(next_tick);
    }
  }

  void publish_joint_states(const RobotState & state)
  {
    if (!joint_state_pub_) {
      return;
    }
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = now();
    msg.name.assign(RB10Model::joint_names.begin(), RB10Model::joint_names.end());
    msg.position.assign(state.q.data(), state.q.data() + state.q.size());
    msg.velocity.assign(state.qd.data(), state.qd.data() + state.qd.size());
    if (rt_joint_state_pub_) {
      rt_joint_state_pub_->tryPublish(msg);
      return;
    }
    joint_state_pub_->publish(msg);
  }

  void publish_direct_command(const RobotState & command_state)
  {
    if (!direct_command_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray command;
    const JointVector & command_vector =
      command_mode_ == CommandMode::kVelocity ? command_state.qd : command_state.q;
    command.data.assign(command_vector.data(), command_vector.data() + command_vector.size());
    if (rt_direct_command_pub_) {
      rt_direct_command_pub_->tryPublish(command);
      return;
    }
    direct_command_pub_->publish(command);
  }

  void publish_position_command(const RobotState & command_state)
  {
    if (!position_command_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray command;
    command.data.assign(command_state.q.data(), command_state.q.data() + command_state.q.size());
    if (rt_position_command_pub_) {
      rt_position_command_pub_->tryPublish(command);
      return;
    }
    position_command_pub_->publish(command);
  }

  void publish_target_q(const RobotState & command_state)
  {
    if (!target_q_pub_) {
      return;
    }

    std_msgs::msg::Float64MultiArray command;
    command.data.assign(command_state.q.data(), command_state.q.data() + command_state.q.size());
    if (rt_target_q_pub_) {
      rt_target_q_pub_->tryPublish(command);
      return;
    }
    target_q_pub_->publish(command);
  }

  void publish_visualization()
  {
    const bool publish_goal_tf_visualization =
      publish_goal_tf_enabled_ && static_cast<bool>(goal_tf_broadcaster_);
    const bool publish_goal_marker_visualization = static_cast<bool>(goal_marker_pub_);
    const bool publish_context_visualization =
      static_cast<bool>(control_point_pub_) ||
      static_cast<bool>(body_obstacle_pub_) ||
      static_cast<bool>(debug_state_pub_) ||
      static_cast<bool>(eef_pose_pub_) ||
      static_cast<bool>(repulsion_metric_pub_) ||
      static_cast<bool>(tcp_accel_pub_) ||
      static_cast<bool>(tangent_escape_debug_pub_);

    if (
      !publish_goal_marker_visualization &&
      !publish_context_visualization &&
      !publish_goal_tf_visualization)
    {
      return;
    }

    if (rmp_flag_gate_enabled_ && (!rmp_active_.load() || !external_goal_received_.load())) {
      if (publish_goal_marker_visualization || publish_context_visualization) {
        clear_rmp_visualization();
      }
      return;
    }

    GoalTarget goal_target;
    goal_target_box_.get(goal_target);
    if (publish_goal_tf_visualization) {
      publish_goal_tf(goal_target);
    }

    const Eigen::Vector3d goal = goal_target.position;
    const auto stamp = now();

    if (goal_marker_pub_) {
      visualization_msgs::msg::Marker goal_marker;
      goal_marker.header.frame_id = "base_link";
      goal_marker.header.stamp = stamp;
      goal_marker.ns = "goal";
      goal_marker.id = 0;
      goal_marker.type = visualization_msgs::msg::Marker::SPHERE;
      goal_marker.action = visualization_msgs::msg::Marker::ADD;
      goal_marker.pose.position.x = goal.x();
      goal_marker.pose.position.y = goal.y();
      goal_marker.pose.position.z = goal.z();
      goal_marker.pose.orientation.w = 1.0;
      goal_marker.scale.x = 0.04;
      goal_marker.scale.y = 0.04;
      goal_marker.scale.z = 0.04;
      goal_marker.color.r = 0.0F;
      goal_marker.color.g = 1.0F;
      goal_marker.color.b = 0.0F;
      goal_marker.color.a = 0.8F;
      goal_marker_pub_->publish(goal_marker);
    }

    if (!publish_context_visualization) {
      visualization_cleared_for_inactive_ = false;
      return;
    }

    const auto state = backend_->read_state();
    const auto context = RB10Model::forward_context(state.q);

    if (debug_state_pub_) {
      publish_debug_state(state, context);
    }

    if (control_point_pub_) {
      visualization_msgs::msg::MarkerArray points;
      for (std::size_t index = 0; index < context.control_points.size(); ++index) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = stamp;
        marker.ns = "control_points";
        marker.id = static_cast<int>(index);
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = context.control_points[index].position.x();
        marker.pose.position.y = context.control_points[index].position.y();
        marker.pose.position.z = context.control_points[index].position.z();
        marker.pose.orientation.w = 1.0;
        const double marker_diameter = control_point_marker_diameter_ > 0.0 ?
          control_point_marker_diameter_ :
          context.control_points[index].radius * 2.0;
        marker.scale.x = marker_diameter;
        marker.scale.y = marker_diameter;
        marker.scale.z = marker_diameter;
        marker.color.r = 0.0F;
        marker.color.g = 0.5F;
        marker.color.b = 1.0F;
        marker.color.a = 0.3F;
        points.markers.push_back(marker);
      }
      control_point_pub_->publish(points);
    }

    if (body_obstacle_pub_) {
      visualization_msgs::msg::MarkerArray body_obstacles;
      for (std::size_t index = 0; index < body_obstacles_visual_.size(); ++index) {
        const auto & obstacle = body_obstacles_visual_[index];
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "base_link";
        marker.header.stamp = stamp;
        marker.ns = "body_obstacles";
        marker.id = static_cast<int>(index);
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.color.r = 1.0F;
        marker.color.g = 0.6F;
        marker.color.b = 0.0F;
        marker.color.a = 0.22F;

        Eigen::Vector3d center = obstacle.center;
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
        if (!obstacle.link_name.empty()) {
          const auto link_index = link_index_from_name(obstacle.link_name);
          rotation = context.link_rotations[link_index];
          if (obstacle.type == "box") {
            center =
              context.link_positions[link_index] +
              rotation * (0.5 * (obstacle.mins + obstacle.maxs));
          } else {
            center =
              context.link_positions[link_index] +
              rotation * obstacle.center;
          }
        } else if (obstacle.type == "box") {
          center = 0.5 * (obstacle.mins + obstacle.maxs);
        }

        marker.pose.position.x = center.x();
        marker.pose.position.y = center.y();
        marker.pose.position.z = center.z();

        if (obstacle.type == "ball") {
          marker.type = visualization_msgs::msg::Marker::SPHERE;
          marker.pose.orientation.w = 1.0;
          marker.scale.x = obstacle.radius * 2.0;
          marker.scale.y = obstacle.radius * 2.0;
          marker.scale.z = obstacle.radius * 2.0;
        } else if (obstacle.type == "box") {
          marker.type = visualization_msgs::msg::Marker::CUBE;
          Eigen::Quaterniond q(rotation);
          q.normalize();
          marker.pose.orientation.x = q.x();
          marker.pose.orientation.y = q.y();
          marker.pose.orientation.z = q.z();
          marker.pose.orientation.w = q.w();
          const Eigen::Vector3d size = obstacle.maxs - obstacle.mins;
          marker.scale.x = size.x();
          marker.scale.y = size.y();
          marker.scale.z = size.z();
        } else {
          continue;
        }

        body_obstacles.markers.push_back(marker);
      }
      body_obstacle_pub_->publish(body_obstacles);
    }

    publish_repulsion_metric_markers(state, context);
    publish_tcp_accel_marker(context);
    publish_tangent_escape_filter_markers();

    if (eef_pose_pub_) {
      geometry_msgs::msg::PoseStamped eef_pose;
      const Eigen::Vector3d & eef = context.tcp_position;
      eef_pose.header.frame_id = "base_link";
      eef_pose.header.stamp = stamp;
      eef_pose.pose.position.x = eef.x();
      eef_pose.pose.position.y = eef.y();
      eef_pose.pose.position.z = eef.z();
      Eigen::Quaterniond tcp_orientation(context.link_rotations[RB10Model::TCP_RMP]);
      tcp_orientation.normalize();
      eef_pose.pose.orientation.x = tcp_orientation.x();
      eef_pose.pose.orientation.y = tcp_orientation.y();
      eef_pose.pose.orientation.z = tcp_orientation.z();
      eef_pose.pose.orientation.w = tcp_orientation.w();
      eef_pose_pub_->publish(eef_pose);
    }
    visualization_cleared_for_inactive_ = false;
  }

  void clear_rmp_visualization()
  {
    if (visualization_cleared_for_inactive_) {
      return;
    }

    visualization_msgs::msg::Marker clear_goal;
    clear_goal.header.frame_id = "base_link";
    clear_goal.header.stamp = now();
    clear_goal.action = visualization_msgs::msg::Marker::DELETEALL;
    if (goal_marker_pub_) {
      goal_marker_pub_->publish(clear_goal);
    }

    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "base_link";
    clear_marker.header.stamp = now();
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;

    visualization_msgs::msg::MarkerArray clear_array;
    clear_array.markers.push_back(clear_marker);
    if (control_point_pub_) {
      control_point_pub_->publish(clear_array);
    }
    if (body_obstacle_pub_) {
      body_obstacle_pub_->publish(clear_array);
    }
    if (repulsion_metric_pub_) {
      repulsion_metric_pub_->publish(clear_array);
    }
    if (tcp_accel_pub_) {
      tcp_accel_pub_->publish(clear_marker);
    }
    if (tangent_escape_debug_pub_) {
      tangent_escape_debug_pub_->publish(clear_array);
    }
    visualization_cleared_for_inactive_ = true;
  }

  std::atomic<bool> running_;
  std::thread control_thread_;
  std::unique_ptr<ControllerBackend> backend_;
  std::unique_ptr<RmpSolverInterface> solver_;
  RobotState state_;
  RobotState virtual_velocity_state_;
  RobotState last_safe_command_state_;
  JointVector controller_velocity_previous_q_{JointVector::Zero()};
  JointVector controller_velocity_estimate_{JointVector::Zero()};
  bool virtual_velocity_state_initialized_{false};
  bool last_safe_command_state_initialized_{false};
  bool controller_velocity_estimator_initialized_{false};
  Eigen::Vector3d body_goal_;
  CommandMode command_mode_{CommandMode::kPosition};
  TargetRmpParams target_metric_params_;
  CollisionRmpParams collision_metric_params_;
  realtime_tools::RealtimeBox<GoalTarget> goal_target_box_;
  realtime_tools::RealtimeBox<std::vector<ObstacleSphere>> obstacles_box_;
  realtime_tools::RealtimeBox<std::vector<std::optional<ObstacleSphere>>>
  proximity_sensor_obstacles_box_;
  realtime_tools::RealtimeBox<TcpAccelerationSample> tcp_accel_visualization_box_;
  realtime_tools::RealtimeBox<TangentEscapeDebugState> tangent_escape_debug_box_;
  std::vector<BodyObstacle> body_obstacles_visual_;
  std::atomic<bool> last_min_z_safety_triggered_{false};
  bool initialize_goal_from_first_state_{true};
  std::atomic<bool> startup_goal_synced_{false};
  std::atomic<bool> external_goal_received_{false};
  mutable std::mutex external_rmp_mutex_;
  std::unordered_map<std::string, ExternalRmpBuffer> external_rmp_buffers_;
  bool publish_joint_states_enabled_{true};
  bool publish_visualization_enabled_{true};
  bool publish_goal_marker_enabled_{true};
  bool publish_control_point_markers_enabled_{true};
  bool publish_body_obstacle_markers_enabled_{true};
  bool publish_end_effector_pose_enabled_{true};
  bool publish_debug_state_enabled_{true};
  bool publish_repulsion_metric_markers_enabled_{true};
  bool publish_tcp_accel_marker_enabled_{true};
  bool publish_rmp_ee_pose_enabled_{true};
  bool publish_goal_tf_enabled_{true};
  double control_point_marker_diameter_{0.10};
  std::string joint_state_topic_{"joint_states"};
  std::string goal_tf_parent_frame_{"base_link"};
  std::string goal_tf_child_frame_{"rmp_goal_target"};
  bool rmp_flag_gate_enabled_{false};
  bool visualization_cleared_for_inactive_{false};
  double min_goal_z_{0.05};
  bool safety_stop_on_min_z_{true};
  double workspace_floor_z_{0.0};
  double min_link6_z_{0.03};
  double min_tcp_z_{0.05};
  double control_rate_hz_{100.0};
  bool enable_realtime_{false};
  int realtime_priority_{80};
  bool lock_memory_{false};
  bool estimate_velocity_in_controller_{false};
  double measured_position_feedback_blend_{1.0};
  double measured_velocity_feedback_blend_{0.35};
  bool use_velocity_feedback_in_solver_{true};
  double controller_velocity_filter_alpha_{0.25};
  double max_joint_accel_{20.0};
  double command_guard_max_step_rad_{0.00436332313};
  double command_guard_max_velocity_rad_s_{0.436332313};
  bool predictive_joint_limit_guard_{true};
  std::array<double, 6> joint_lower_limits_ = RB10Model::joint_lower_limits;
  std::array<double, 6> joint_upper_limits_ = RB10Model::joint_upper_limits;
  std::array<double, 6> joint_limit_buffers_{0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
  double repulsion_metric_marker_min_norm_{0.01};
  double repulsion_metric_marker_dot_diameter_{0.04};
  double tcp_accel_marker_max_length_{0.15};
  double tcp_accel_marker_norm_for_max_length_{2.0};
  double tcp_accel_marker_min_norm_{0.001};
  bool enable_tangent_escape_filter_{false};
  bool publish_tangent_escape_filter_debug_enabled_{false};
  bool publish_tangent_escape_filter_data_enabled_{false};
  bool publish_tangent_escape_filter_candidate_data_enabled_{false};
  double tangent_escape_safe_distance_{0.08};
  double tangent_escape_influence_distance_{0.25};
  double tangent_escape_tangent_gain_{2.0};
  double tangent_escape_normal_gain_{0.0};
  double tangent_escape_damping_{0.02};
  double tangent_escape_max_delta_qdd_{8.0};
  int tangent_escape_candidate_count_{16};
  double tangent_escape_candidate_lookahead_{0.08};
  double tangent_escape_goal_weight_{1.0};
  double tangent_escape_continuity_weight_{0.5};
  double tangent_escape_up_weight_{0.0};
  double tangent_escape_duplicate_risk_weight_{2.0};
  double tangent_escape_adjacent_block_weight_{1.0};
  double tangent_escape_branch_hold_weight_{0.5};
  double tangent_escape_min_activation_{0.05};
  double tangent_escape_min_tangent_norm_{1e-4};
  double tangent_escape_goal_normal_dot_threshold_{1.0};
  double tangent_escape_marker_length_{0.12};
  Eigen::Vector3d tangent_escape_previous_tangent_{Eigen::Vector3d::UnitX()};
  std::size_t tangent_escape_previous_control_point_index_{0};
  bool tangent_escape_previous_tangent_valid_{false};
  int rmp_active_flag_value_{1};
  std::atomic<bool> rmp_active_{true};

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr direct_command_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr position_command_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_q_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_metric_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr rmp_joint_accel_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr rmp_tcp_accel_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr tangent_escape_filter_data_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr
    tangent_escape_filter_candidate_data_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr control_point_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr body_obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr repulsion_metric_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tangent_escape_debug_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tcp_accel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr eef_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr rmp_eef_pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr debug_state_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> goal_tf_broadcaster_;
  std::shared_ptr<JointStateRtPublisher> rt_joint_state_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_direct_command_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_position_command_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_target_q_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_target_metric_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_rmp_joint_accel_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_rmp_tcp_accel_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_tangent_escape_filter_data_pub_;
  std::shared_ptr<Float64ArrayRtPublisher> rt_tangent_escape_filter_candidate_data_pub_;
  std::shared_ptr<PoseRtPublisher> rt_rmp_eef_pose_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr rmp_flag_sub_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr obstacle_sub_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr> external_metric_subs_;
  std::vector<rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr> external_accel_subs_;
  rclcpp::TimerBase::SharedPtr visualization_timer_;
};

}  // namespace

}  // namespace rb10_rmpflow_rviz

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rb10_rmpflow_rviz::RmpflowControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

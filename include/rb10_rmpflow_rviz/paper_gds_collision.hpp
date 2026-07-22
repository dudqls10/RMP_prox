#pragma once

namespace rb10_rmpflow_rviz
{
namespace paper_gds_collision
{

// Parameters for a signed-clearance, one-dimensional structured GDS collision
// leaf.  The names intentionally mirror the existing collision RMP parameters
// where their physical meaning is unchanged.
struct Params
{
  double metric_scalar{1.0};
  double metric_modulation_radius{0.3};
  double metric_exploder_std_dev{0.02};
  double metric_exploder_eps{0.8};
  double clearance_smoothing{1.0e-4};

  // u(s_dot) is bounded below by this value.  A positive floor keeps the leaf
  // non-degenerate at rest while retaining the RMPflow condition
  // s_dot * du/ds_dot >= 0.
  double metric_velocity_floor{0.5};
  double metric_velocity_scale{0.03};

  // repulsion_gain remains a desired acceleration at rest.  The implementation
  // converts it to the gradient of a position-only potential.
  double repulsion_gain{500.0};
  double repulsion_std_dev{0.05};

  double damping_gain{50.0};
  double damping_std_dev{0.01};
  double damping_robustness_eps{0.01};
  double damping_velocity_scale{0.03};
};

struct Result
{
  double signed_clearance{0.0};
  double clearance_rate{0.0};

  double smooth_positive_clearance{0.0};
  double smooth_positive_clearance_derivative{0.0};
  double distance_gate{0.0};
  double distance_gate_derivative{0.0};

  double w{0.0};
  double w_derivative{0.0};
  double u{0.0};
  double u_derivative{0.0};
  double clearance_rate_times_u_derivative{0.0};

  // Structured-GDS quantities:
  //   G = w u
  //   Xi = 0.5 w s_dot u'
  //   xi = 0.5 u w' s_dot^2
  //   M = G + Xi
  double G{0.0};
  double Xi{0.0};
  double xi{0.0};
  double M{0.0};

  // potential_force is -dPhi/ds.  B is the non-negative damping
  // coefficient, so damping_force = -B s_dot.
  double repulsion_acceleration{0.0};
  double potential_force{0.0};
  double damping_gate{0.0};
  double B{0.0};
  double damping_force{0.0};
  double natural_force{0.0};

  bool finite{false};
  bool metric_psd{false};
  bool damping_psd{false};
  bool theorem_condition{false};
  bool distance_metric_nonincreasing{false};
  bool valid{false};
};

// Checks parameter domains without throwing.
bool parameters_are_valid(const Params & params) noexcept;

// Evaluates the natural-form leaf [natural_force, M] for signed clearance s
// and clearance rate s_dot.  Invalid parameters or non-finite inputs raise
// std::invalid_argument.
Result evaluate(
  double signed_clearance,
  double clearance_rate,
  const Params & params);

}  // namespace paper_gds_collision
}  // namespace rb10_rmpflow_rviz

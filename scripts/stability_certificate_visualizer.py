#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker


class StabilityCertificateVisualizer(Node):
    """Render the numerical/conditional stability certificate as RViz text."""

    def __init__(self) -> None:
        super().__init__("stability_certificate_visualizer")
        self.declare_parameter(
            "certificate_topic",
            "/rmp_stability_certificate",
        )
        self.declare_parameter(
            "marker_topic",
            "/rmp_stability_certificate_marker",
        )
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", 1.55)
        self.declare_parameter("text_height", 0.055)

        self.publisher = self.create_publisher(
            Marker,
            str(self.get_parameter("marker_topic").value),
            10,
        )
        self.subscription = self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("certificate_topic").value),
            self._on_certificate,
            10,
        )

    @staticmethod
    def _value(values, index, default=0.0):
        if index >= len(values):
            return default
        value = float(values[index])
        return value if math.isfinite(value) else default

    def _on_certificate(self, msg: Float64MultiArray) -> None:
        values = list(msg.data)
        if len(values) < 31 or int(round(self._value(values, 0))) != 1:
            return

        structural = self._value(values, 1) > 0.5
        environment_static = self._value(values, 2) > 0.5
        guard_enabled = self._value(values, 4) > 0.5
        certified = self._value(values, 5) > 0.5
        scale = self._value(values, 6)
        tank = self._value(values, 7)
        capacity = self._value(values, 8)
        requested_power = self._value(values, 9)
        applied_power = self._value(values, 10)
        numerical_power = self._value(values, 13)
        energy_violation = self._value(values, 28)
        config_profile = self._value(values, 31, float(structural)) > 0.5
        base_domain = self._value(values, 32, float(structural)) > 0.5
        base_metric_spd = self._value(values, 33, float(structural)) > 0.5
        base_solve = self._value(values, 34, float(structural)) > 0.5

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = str(self.get_parameter("frame_id").value)
        marker.ns = "rmp_stability_certificate"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.get_parameter("x").value)
        marker.pose.position.y = float(self.get_parameter("y").value)
        marker.pose.position.z = float(self.get_parameter("z").value)
        marker.pose.orientation.w = 1.0
        marker.scale.z = float(self.get_parameter("text_height").value)
        marker.color.a = 1.0

        if certified and energy_violation <= 1e-9:
            marker.color.r = 0.15
            marker.color.g = 1.0
            marker.color.b = 0.25
            verdict = "MODEL-RATE CERTIFICATE: PASS"
        elif guard_enabled and energy_violation <= 1e-9:
            marker.color.r = 1.0
            marker.color.g = 0.78
            marker.color.b = 0.05
            verdict = "SAMPLED ESCAPE BUDGET: PASS / MODEL RATE: N/A"
        else:
            marker.color.r = 1.0
            marker.color.g = 0.15
            marker.color.b = 0.10
            verdict = "CERTIFICATE: NOT SATISFIED"

        marker.text = (
            f"{verdict}\n"
            f"base GDS={int(structural)}  env static={int(environment_static)}  "
            f"guard={int(guard_enabled)}\n"
            f"cfg/domain/SPD/solve={int(config_profile)}/{int(base_domain)}/"
            f"{int(base_metric_spd)}/{int(base_solve)}\n"
            f"Pesc req/applied={requested_power:+.3e}/{applied_power:+.3e}  "
            f"scale={scale:.3f}\n"
            f"tank={tank:.4g}/{capacity:.4g}  "
            f"Psolve+clamp={numerical_power:+.3e}\n"
            f"energy-bound violation={energy_violation:.3e}\n"
            "does not certify W[k+1] or hard collision safety"
        )
        self.publisher.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StabilityCertificateVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

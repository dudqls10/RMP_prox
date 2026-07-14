#!/usr/bin/env python3
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import UInt8


SENSOR_TOPIC_MAP = [
    ("tof6_1_L", 1),
    ("tof6_1_F", 2),
    ("tof6_1_R", 3),
    ("tof6_1_U", 4),
    ("tof_S", 6),
    ("tof_E", 7),
    ("tof_N", 8),
    ("tof_W", 5),
    ("tof3_1_S", 12),
    ("tof3_1_W", 9),
    ("tof3_1_N", 10),
    ("tof3_1_E", 11),
    ("tof2_N", 13),
    ("tof2_W", 14),
    ("tof2_S", 15),
    ("tof2_E", 16),
    ("tof2_1_E", 17),
    ("tof2_1_S", 18),
    ("tof2_1_W", 19),
    ("tof2_1_N", 20),
]

RANDOM_SCENARIOS = {"random", "random_approach_retreat"}
RANDOM_PULSE_SCENARIOS = {"random_pulse"}


@dataclass(frozen=True)
class RandomSensorEvent:
    index: int
    start_s: float
    end_s: float
    sensor_names: Tuple[str, ...]


class FakeProximityScenario(Node):
    def __init__(self) -> None:
        super().__init__("fake_proximity_scenario")

        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("scenario", "single")
        self.declare_parameter("sensor_name", "tof6_1_R")
        self.declare_parameter("active_sensor_names_csv", "")
        self.declare_parameter("start_s", 3.0)
        self.declare_parameter("duration_s", 8.0)
        self.declare_parameter("period_s", 10.0)
        self.declare_parameter("hold_s", 3.0)
        self.declare_parameter("range_m", 0.10)
        self.declare_parameter("inactive_range_m", 0.90)
        self.declare_parameter("approach_start_range_m", 0.35)
        self.declare_parameter("approach_end_range_m", 0.50)
        self.declare_parameter("random_count", 5)
        self.declare_parameter("random_seed", 1)
        self.declare_parameter("random_sensor_count", 1)
        self.declare_parameter("random_allow_repeats", False)
        self.declare_parameter("range_scale", 0.001)
        self.declare_parameter("max_raw_range", 1000.0)
        self.declare_parameter("field_of_view", 0.12)
        self.declare_parameter("proximity_topic_prefix", "proximity_distance")
        self.declare_parameter("raw_topic_prefix", "raw_distance")
        self.declare_parameter("publish_raw_topics", True)
        self.declare_parameter("publish_rmp_flag", True)
        self.declare_parameter("rmp_flag_topic", "/RMP_flag")
        self.declare_parameter("rmp_active_flag_value", 1)

        self.publish_rate_hz = max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.scenario = str(self.get_parameter("scenario").value).strip().lower()
        self.sensor_name = str(self.get_parameter("sensor_name").value).strip()
        self.start_s = max(float(self.get_parameter("start_s").value), 0.0)
        self.duration_s = float(self.get_parameter("duration_s").value)
        self.period_s = max(float(self.get_parameter("period_s").value), 1e-3)
        self.hold_s = max(float(self.get_parameter("hold_s").value), 0.0)
        self.range_m = max(float(self.get_parameter("range_m").value), 0.0)
        self.inactive_range_m = max(float(self.get_parameter("inactive_range_m").value), 0.0)
        self.approach_start_range_m = max(
            float(self.get_parameter("approach_start_range_m").value),
            self.range_m,
        )
        self.approach_end_range_m = max(
            float(self.get_parameter("approach_end_range_m").value),
            self.range_m,
        )
        self.random_count = max(int(self.get_parameter("random_count").value), 0)
        self.random_seed = int(self.get_parameter("random_seed").value)
        self.random_sensor_count = max(int(self.get_parameter("random_sensor_count").value), 1)
        self.random_allow_repeats = self._as_bool(
            self.get_parameter("random_allow_repeats").value
        )
        self.range_scale = max(float(self.get_parameter("range_scale").value), 1e-9)
        self.max_raw_range = max(float(self.get_parameter("max_raw_range").value), 1.0)
        self.field_of_view = max(float(self.get_parameter("field_of_view").value), 0.0)
        self.proximity_topic_prefix = str(
            self.get_parameter("proximity_topic_prefix").value
        ).strip()
        self.raw_topic_prefix = str(self.get_parameter("raw_topic_prefix").value).strip()
        self.publish_raw_topics = self._as_bool(self.get_parameter("publish_raw_topics").value)
        self.publish_rmp_flag = self._as_bool(self.get_parameter("publish_rmp_flag").value)
        self.rmp_active_flag_value = int(self.get_parameter("rmp_active_flag_value").value)

        self.active_sensor_names = self._active_sensor_names()
        self.proximity_publishers: Dict[str, object] = {}
        self.raw_publishers: Dict[str, object] = {}
        for sensor_name, topic_index in SENSOR_TOPIC_MAP:
            proximity_topic = self._indexed_topic(self.proximity_topic_prefix, topic_index)
            self.proximity_publishers[sensor_name] = self.create_publisher(
                Range,
                proximity_topic,
                10,
            )
            if self.publish_raw_topics:
                raw_topic = self._indexed_topic(self.raw_topic_prefix, topic_index)
                self.raw_publishers[sensor_name] = self.create_publisher(Range, raw_topic, 10)

        self.rmp_flag_pub = None
        if self.publish_rmp_flag:
            self.rmp_flag_pub = self.create_publisher(
                UInt8,
                str(self.get_parameter("rmp_flag_topic").value),
                10,
            )

        self._unknown_scenario_warned = False
        self.random_events = self._make_random_events()
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish)
        self.get_logger().info(
            "fake proximity scenario started: "
            f"scenario={self.scenario}, active={','.join(self.active_sensor_names)}, "
            f"range_m={self.range_m:.3f}, inactive_range_m={self.inactive_range_m:.3f}"
        )
        if self.random_events:
            preview = "; ".join(
                f"{event.index}:{','.join(event.sensor_names)}@"
                f"{event.start_s:.2f}-{event.end_s:.2f}s"
                for event in self.random_events[:10]
            )
            suffix = " ..." if len(self.random_events) > 10 else ""
            self.get_logger().info(
                "random fake proximity events: "
                f"count={len(self.random_events)}, seed={self.random_seed}, "
                f"sensor_count={self.random_sensor_count}, "
                f"period_s={self.period_s:.3f}, hold_s={self.hold_s:.3f}: "
                f"{preview}{suffix}"
            )

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _indexed_topic(prefix: str, index: int) -> str:
        if "{index}" in prefix:
            return prefix.format(index=index)
        return f"{prefix}{index}"

    def _active_sensor_names(self) -> List[str]:
        csv_value = str(self.get_parameter("active_sensor_names_csv").value).strip()
        names = [name.strip() for name in csv_value.split(",") if name.strip()]
        if not names:
            if self.scenario in RANDOM_SCENARIOS or self.scenario in RANDOM_PULSE_SCENARIOS:
                names = [sensor_name for sensor_name, _ in SENSOR_TOPIC_MAP]
            else:
                names = [self.sensor_name]
        if len(names) == 1 and names[0].lower() in {"all", "*"}:
            return [sensor_name for sensor_name, _ in SENSOR_TOPIC_MAP]
        known = {sensor_name for sensor_name, _ in SENSOR_TOPIC_MAP}
        valid = [name for name in names if name in known]
        invalid = sorted(set(names) - known)
        if invalid:
            self.get_logger().warn(
                "Ignoring unknown fake proximity sensor names: " + ",".join(invalid)
            )
        if not valid:
            valid = ["tof6_1_R"]
        return valid

    def _make_random_events(self) -> List[RandomSensorEvent]:
        if self.scenario not in RANDOM_SCENARIOS and self.scenario not in RANDOM_PULSE_SCENARIOS:
            return []
        if self.random_count <= 0 or not self.active_sensor_names:
            return []

        seed = self.random_seed if self.random_seed >= 0 else time.time_ns()
        rng = random.Random(seed)
        pool = list(self.active_sensor_names)
        sensor_count = min(self.random_sensor_count, len(pool))
        bag: List[str] = []
        events: List[RandomSensorEvent] = []
        hold_s = max(self.hold_s, 1.0 / self.publish_rate_hz)

        for index in range(self.random_count):
            if self.random_allow_repeats:
                selected = tuple(rng.sample(pool, sensor_count))
            else:
                selected_names: List[str] = []
                while len(selected_names) < sensor_count:
                    if not bag:
                        bag = pool.copy()
                        rng.shuffle(bag)
                    selected_names.append(bag.pop())
                selected = tuple(selected_names)

            start_s = self.start_s + index * self.period_s
            events.append(
                RandomSensorEvent(
                    index=index,
                    start_s=start_s,
                    end_s=start_s + hold_s,
                    sensor_names=selected,
                )
            )
        return events

    def _elapsed_s(self) -> float:
        delta = self.get_clock().now() - self.start_time
        return float(delta.nanoseconds) * 1e-9

    def _inside_window(self, elapsed_s: float) -> bool:
        if elapsed_s < self.start_s:
            return False
        if self.duration_s <= 0.0:
            return True
        return elapsed_s <= self.start_s + self.duration_s

    def _approach_hold_retreat_range(self, phase_s: float, total_s: float) -> float:
        total_s = max(total_s, 1e-6)
        phase_s = min(max(phase_s, 0.0), total_s)
        hold_s = min(self.hold_s, total_s)
        ramp_s = max((total_s - hold_s) * 0.5, 0.0)
        if ramp_s <= 1e-6:
            return self.range_m
        if phase_s <= ramp_s:
            alpha = phase_s / ramp_s
            return self.approach_start_range_m + alpha * (
                self.range_m - self.approach_start_range_m
            )
        if phase_s <= ramp_s + hold_s:
            return self.range_m
        alpha = min(max((phase_s - ramp_s - hold_s) / ramp_s, 0.0), 1.0)
        return self.range_m + alpha * (self.approach_end_range_m - self.range_m)

    def _current_active_set(self, elapsed_s: float) -> Set[str]:
        if self.scenario in {"off", "none"}:
            return set()
        if self.scenario in RANDOM_SCENARIOS or self.scenario in RANDOM_PULSE_SCENARIOS:
            active: Set[str] = set()
            for event in self.random_events:
                if event.start_s <= elapsed_s <= event.end_s:
                    active.update(event.sensor_names)
            return active
        if self.scenario in {"single", "wall", "approach", "approach_retreat"}:
            return set(self.active_sensor_names) if self._inside_window(elapsed_s) else set()
        if self.scenario == "pulse":
            if elapsed_s < self.start_s:
                return set()
            phase = math.fmod(elapsed_s - self.start_s, self.period_s)
            return set(self.active_sensor_names) if phase <= self.hold_s else set()
        if self.scenario == "cycle":
            if elapsed_s < self.start_s or not self.active_sensor_names:
                return set()
            phase_time = elapsed_s - self.start_s
            index = int(phase_time // self.period_s) % len(self.active_sensor_names)
            phase = math.fmod(phase_time, self.period_s)
            if phase <= self.hold_s:
                return {self.active_sensor_names[index]}
            return set()
        if not self._unknown_scenario_warned:
            self.get_logger().warn(
                f"Unknown scenario '{self.scenario}', falling back to single"
            )
            self._unknown_scenario_warned = True
        return set(self.active_sensor_names) if self._inside_window(elapsed_s) else set()

    def _active_range_m(self, elapsed_s: float) -> float:
        if self.scenario not in {"approach", "approach_retreat"} or self.duration_s <= 1e-6:
            return self.range_m

        if self.scenario == "approach_retreat":
            return self._approach_hold_retreat_range(
                elapsed_s - self.start_s,
                self.duration_s,
            )

        alpha = min(max((elapsed_s - self.start_s) / self.duration_s, 0.0), 1.0)
        return self.approach_start_range_m + alpha * (self.range_m - self.approach_start_range_m)

    def _random_event_range_m(self, event: RandomSensorEvent, elapsed_s: float) -> float:
        if self.scenario in RANDOM_PULSE_SCENARIOS:
            return self.range_m
        event_duration = max(event.end_s - event.start_s, 1e-6)
        return self._approach_hold_retreat_range(
            elapsed_s - event.start_s,
            event_duration,
        )

    def _sensor_range_m(self, sensor_name: str, elapsed_s: float) -> float:
        if self.scenario in RANDOM_SCENARIOS or self.scenario in RANDOM_PULSE_SCENARIOS:
            ranges = [
                self._random_event_range_m(event, elapsed_s)
                for event in self.random_events
                if event.start_s <= elapsed_s <= event.end_s and sensor_name in event.sensor_names
            ]
            return min(ranges) if ranges else self.inactive_range_m

        active_sensors = self._current_active_set(elapsed_s)
        if sensor_name not in active_sensors:
            return self.inactive_range_m
        return self._active_range_m(elapsed_s)

    def _make_range_msg(self, sensor_name: str, range_m: float) -> Range:
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = sensor_name
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = self.field_of_view
        msg.min_range = 0.0
        msg.max_range = self.max_raw_range
        msg.range = min(max(range_m / self.range_scale, 0.0), self.max_raw_range - 1.0)
        return msg

    def _publish(self) -> None:
        elapsed_s = self._elapsed_s()
        for sensor_name, _ in SENSOR_TOPIC_MAP:
            range_m = self._sensor_range_m(sensor_name, elapsed_s)
            msg = self._make_range_msg(sensor_name, range_m)
            self.proximity_publishers[sensor_name].publish(msg)
            raw_pub = self.raw_publishers.get(sensor_name)
            if raw_pub is not None:
                raw_pub.publish(msg)

        if self.rmp_flag_pub is not None:
            flag = UInt8()
            flag.data = self.rmp_active_flag_value
            self.rmp_flag_pub.publish(flag)


def main() -> None:
    rclpy.init()
    node = FakeProximityScenario()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import bisect
import csv
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class CsvReplayData:
    path: str
    metadata: Dict[str, str]
    timestamps_s: Tuple[float, ...]
    ranges_m: Tuple[Tuple[Optional[float], ...], ...]
    input_rate_hz: float
    valid_value_count: int
    invalid_value_count: int


def _read_csv_lines(path: str) -> Tuple[Dict[str, str], List[str]]:
    metadata: Dict[str, str] = {}
    data_lines: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as stream:
        for line in stream:
            if line.startswith("#"):
                key_value = line[1:].strip().split(",", 1)
                if len(key_value) == 2:
                    metadata[key_value[0].strip()] = key_value[1].strip()
                continue
            if line.strip():
                data_lines.append(line)
    return metadata, data_lines


def _unit_scale_to_m(input_unit: str, metadata: Dict[str, str], column_prefix: str) -> float:
    normalized = input_unit.strip().lower()
    if normalized == "auto":
        normalized = metadata.get("range_unit", "").strip().lower()
        if not normalized and "proximity_distance" in column_prefix:
            normalized = "millimeters"

    aliases = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "metre": 1.0,
        "metres": 1.0,
        "mm": 0.001,
        "millimeter": 0.001,
        "millimeters": 0.001,
        "millimetre": 0.001,
        "millimetres": 0.001,
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized in {"raw", "sensor_raw", "message_range", ""}:
        raise ValueError(
            "CSV input unit is not a physical distance. Record /proximity_distance1..20 "
            "with range_unit=millimeters, or set input_unit explicitly to meters/millimeters."
        )
    raise ValueError(
        f"Unsupported input_unit '{input_unit}'. Use auto, meters, or millimeters."
    )


def load_replay_csv(
    path: str,
    timestamp_column: str,
    distance_column_prefix: str,
    input_unit: str,
    minimum_valid_range_m: float,
    maximum_valid_range_m: float,
) -> CsvReplayData:
    expanded_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(expanded_path):
        raise ValueError(f"CSV file does not exist: {expanded_path}")

    metadata, data_lines = _read_csv_lines(expanded_path)
    if not data_lines:
        raise ValueError(f"CSV has no header or data rows: {expanded_path}")

    reader = csv.DictReader(data_lines)
    fieldnames = list(reader.fieldnames or [])
    if timestamp_column not in fieldnames:
        raise ValueError(
            f"CSV is missing timestamp column '{timestamp_column}': {expanded_path}"
        )

    expected_columns = [f"{distance_column_prefix}{index}" for index in range(1, 21)]
    missing_columns = [name for name in expected_columns if name not in fieldnames]
    if missing_columns:
        raw_columns = [f"raw_distance{index}" for index in range(1, 21)]
        if all(name in fieldnames for name in raw_columns):
            raise ValueError(
                "CSV contains raw_distance1..20, not physical proximity distances. "
                "The approximately 40-million raw values cannot be replayed as millimeters. "
                "Record /proximity_distance1..20 and use that CSV instead."
            )
        raise ValueError(
            "CSV is missing distance columns: " + ", ".join(missing_columns)
        )

    if "raw" in distance_column_prefix.strip().lower():
        raise ValueError(
            "Raw sensor columns are not accepted as distances. Use proximity_distance1..20."
        )

    scale_to_m = _unit_scale_to_m(input_unit, metadata, distance_column_prefix)
    parsed_rows: List[Tuple[float, Tuple[Optional[float], ...]]] = []
    valid_value_count = 0
    invalid_value_count = 0

    for row in reader:
        try:
            timestamp = float(row.get(timestamp_column, ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(timestamp):
            continue

        ranges: List[Optional[float]] = []
        for column in expected_columns:
            try:
                raw_value = float(row.get(column, ""))
            except (TypeError, ValueError):
                raw_value = float("nan")

            range_m = raw_value * scale_to_m
            if (
                math.isfinite(range_m)
                and range_m > 0.0
                and minimum_valid_range_m <= range_m <= maximum_valid_range_m
            ):
                ranges.append(range_m)
                valid_value_count += 1
            else:
                ranges.append(None)
                invalid_value_count += 1
        parsed_rows.append((timestamp, tuple(ranges)))

    if not parsed_rows:
        raise ValueError(f"CSV has no rows with valid timestamps: {expanded_path}")
    if valid_value_count == 0:
        raise ValueError(
            "CSV has no valid physical distance values after unit conversion. "
            f"Expected {minimum_valid_range_m:g}..{maximum_valid_range_m:g} m."
        )

    parsed_rows.sort(key=lambda item: item[0])
    unique_rows: List[Tuple[float, Tuple[Optional[float], ...]]] = []
    for item in parsed_rows:
        if unique_rows and item[0] <= unique_rows[-1][0]:
            if item[0] == unique_rows[-1][0]:
                unique_rows[-1] = item
            continue
        unique_rows.append(item)

    if len(unique_rows) < 2:
        raise ValueError("CSV replay requires at least two rows with increasing timestamps.")

    first_timestamp = unique_rows[0][0]
    timestamps_s = tuple(item[0] - first_timestamp for item in unique_rows)
    ranges_m = tuple(item[1] for item in unique_rows)
    positive_steps = [
        timestamps_s[index] - timestamps_s[index - 1]
        for index in range(1, len(timestamps_s))
        if timestamps_s[index] > timestamps_s[index - 1]
    ]
    median_step = statistics.median(positive_steps)
    input_rate_hz = 1.0 / median_step if median_step > 0.0 else 100.0

    return CsvReplayData(
        path=expanded_path,
        metadata=metadata,
        timestamps_s=timestamps_s,
        ranges_m=ranges_m,
        input_rate_hz=input_rate_hz,
        valid_value_count=valid_value_count,
        invalid_value_count=invalid_value_count,
    )


class FakeProximityCsvReplay(Node):
    def __init__(self) -> None:
        super().__init__("fake_proximity_scenario_v2")

        self.declare_parameter("csv_path", "")
        self.declare_parameter("timestamp_column", "timestamp_unix")
        self.declare_parameter("distance_column_prefix", "proximity_distance")
        self.declare_parameter("input_unit", "auto")
        self.declare_parameter("playback_rate", 1.0)
        self.declare_parameter("publish_rate_hz", 0.0)
        self.declare_parameter("start_offset_s", 0.0)
        self.declare_parameter("duration_s", 0.0)
        self.declare_parameter("start_delay_s", 1.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("inactive_range_m", 0.90)
        self.declare_parameter("minimum_valid_range_m", 0.001)
        self.declare_parameter("maximum_valid_range_m", 10.0)
        self.declare_parameter("output_range_scale", 0.001)
        self.declare_parameter("max_raw_range", 2000.0)
        self.declare_parameter("field_of_view", 0.12)
        self.declare_parameter("proximity_topic_prefix", "/fake_proximity_distance")
        self.declare_parameter("raw_topic_prefix", "/fake_raw_distance")
        self.declare_parameter("publish_raw_topics", True)
        self.declare_parameter("publish_rmp_flag", True)
        self.declare_parameter("rmp_flag_topic", "/RMP_flag")
        self.declare_parameter("rmp_active_flag_value", 1)

        csv_path = str(self.get_parameter("csv_path").value).strip()
        if not csv_path:
            raise ValueError("csv_path is required")

        self.playback_rate = float(self.get_parameter("playback_rate").value)
        if not math.isfinite(self.playback_rate) or self.playback_rate <= 0.0:
            raise ValueError("playback_rate must be greater than zero")

        self.start_offset_s = max(float(self.get_parameter("start_offset_s").value), 0.0)
        self.duration_s = max(float(self.get_parameter("duration_s").value), 0.0)
        self.start_delay_s = max(float(self.get_parameter("start_delay_s").value), 0.0)
        self.loop = self._as_bool(self.get_parameter("loop").value)
        self.inactive_range_m = max(
            float(self.get_parameter("inactive_range_m").value), 0.0
        )
        self.output_range_scale = max(
            float(self.get_parameter("output_range_scale").value), 1e-12
        )
        self.max_raw_range = max(float(self.get_parameter("max_raw_range").value), 1.0)
        self.field_of_view = max(float(self.get_parameter("field_of_view").value), 0.0)
        self.proximity_topic_prefix = str(
            self.get_parameter("proximity_topic_prefix").value
        ).strip()
        self.raw_topic_prefix = str(self.get_parameter("raw_topic_prefix").value).strip()
        self.publish_raw_topics = self._as_bool(
            self.get_parameter("publish_raw_topics").value
        )
        self.publish_rmp_flag = self._as_bool(
            self.get_parameter("publish_rmp_flag").value
        )
        self.rmp_active_flag_value = int(
            self.get_parameter("rmp_active_flag_value").value
        )

        minimum_valid_range_m = max(
            float(self.get_parameter("minimum_valid_range_m").value), 0.0
        )
        maximum_valid_range_m = float(
            self.get_parameter("maximum_valid_range_m").value
        )
        if maximum_valid_range_m <= minimum_valid_range_m:
            raise ValueError("maximum_valid_range_m must exceed minimum_valid_range_m")
        self.max_raw_range = max(
            self.max_raw_range,
            self.inactive_range_m / self.output_range_scale + 2.0,
            maximum_valid_range_m / self.output_range_scale + 2.0,
        )

        self.replay_data = load_replay_csv(
            path=csv_path,
            timestamp_column=str(self.get_parameter("timestamp_column").value).strip(),
            distance_column_prefix=str(
                self.get_parameter("distance_column_prefix").value
            ).strip(),
            input_unit=str(self.get_parameter("input_unit").value),
            minimum_valid_range_m=minimum_valid_range_m,
            maximum_valid_range_m=maximum_valid_range_m,
        )

        last_timestamp_s = self.replay_data.timestamps_s[-1]
        if self.start_offset_s >= last_timestamp_s:
            raise ValueError(
                f"start_offset_s={self.start_offset_s:g} exceeds CSV duration "
                f"{last_timestamp_s:g} s"
            )
        self.window_start_s = self.start_offset_s
        requested_end_s = (
            self.window_start_s + self.duration_s if self.duration_s > 0.0 else last_timestamp_s
        )
        self.window_end_s = min(requested_end_s, last_timestamp_s)
        self.window_duration_s = self.window_end_s - self.window_start_s
        if self.window_duration_s <= 0.0:
            raise ValueError("Selected CSV replay window has zero duration")

        configured_publish_rate = float(self.get_parameter("publish_rate_hz").value)
        if configured_publish_rate > 0.0:
            self.publish_rate_hz = configured_publish_rate
        else:
            self.publish_rate_hz = self.replay_data.input_rate_hz * self.playback_rate
        self.publish_rate_hz = min(max(self.publish_rate_hz, 1.0), 1000.0)

        self.proximity_publishers: Dict[int, object] = {}
        self.raw_publishers: Dict[int, object] = {}
        self.sensor_frames: Dict[int, str] = {}
        for sensor_name, topic_index in SENSOR_TOPIC_MAP:
            self.sensor_frames[topic_index] = sensor_name
            self.proximity_publishers[topic_index] = self.create_publisher(
                Range,
                self._indexed_topic(self.proximity_topic_prefix, topic_index),
                10,
            )
            if self.publish_raw_topics:
                self.raw_publishers[topic_index] = self.create_publisher(
                    Range,
                    self._indexed_topic(self.raw_topic_prefix, topic_index),
                    10,
                )

        self.rmp_flag_pub = None
        if self.publish_rmp_flag:
            self.rmp_flag_pub = self.create_publisher(
                UInt8,
                str(self.get_parameter("rmp_flag_topic").value),
                10,
            )

        self.start_time = self.get_clock().now()
        self.finished_logged = False
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish)

        invalid_ratio = self.replay_data.invalid_value_count / max(
            self.replay_data.valid_value_count + self.replay_data.invalid_value_count,
            1,
        )
        self.get_logger().info(
            "CSV proximity replay ready: "
            f"path={self.replay_data.path}, rows={len(self.replay_data.timestamps_s)}, "
            f"input_rate={self.replay_data.input_rate_hz:.2f} Hz, "
            f"publish_rate={self.publish_rate_hz:.2f} Hz, playback_rate={self.playback_rate:.3f}, "
            f"window={self.window_start_s:.3f}-{self.window_end_s:.3f} s, "
            f"loop={self.loop}, invalid_values={invalid_ratio:.1%}"
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

    def _elapsed_s(self) -> float:
        delta = self.get_clock().now() - self.start_time
        return float(delta.nanoseconds) * 1e-9

    def _make_range_msg(self, topic_index: int, range_m: float) -> Range:
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.sensor_frames[topic_index]
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = self.field_of_view
        msg.min_range = 0.0
        msg.max_range = self.max_raw_range
        raw_range = range_m / self.output_range_scale
        msg.range = min(max(raw_range, 0.0), self.max_raw_range - 1.0)
        return msg

    def _publish_ranges(self, ranges_m: Sequence[Optional[float]], active: bool) -> None:
        for topic_index in range(1, 21):
            source_value = ranges_m[topic_index - 1] if active else None
            range_m = source_value if source_value is not None else self.inactive_range_m
            msg = self._make_range_msg(topic_index, range_m)
            self.proximity_publishers[topic_index].publish(msg)
            raw_publisher = self.raw_publishers.get(topic_index)
            if raw_publisher is not None:
                raw_publisher.publish(msg)

        if self.rmp_flag_pub is not None:
            flag = UInt8()
            flag.data = self.rmp_active_flag_value if active else 0
            self.rmp_flag_pub.publish(flag)

    def _publish(self) -> None:
        elapsed_s = self._elapsed_s()
        if elapsed_s < self.start_delay_s:
            self._publish_ranges((), active=False)
            return

        playback_elapsed_s = (elapsed_s - self.start_delay_s) * self.playback_rate
        if self.loop:
            window_position_s = math.fmod(playback_elapsed_s, self.window_duration_s)
        elif playback_elapsed_s > self.window_duration_s:
            self._publish_ranges((), active=False)
            if not self.finished_logged:
                self.get_logger().info("CSV proximity replay finished; publishing inactive ranges")
                self.finished_logged = True
            return
        else:
            window_position_s = playback_elapsed_s

        replay_timestamp_s = self.window_start_s + window_position_s
        sample_index = bisect.bisect_right(
            self.replay_data.timestamps_s,
            replay_timestamp_s,
        ) - 1
        sample_index = min(max(sample_index, 0), len(self.replay_data.ranges_m) - 1)
        self._publish_ranges(self.replay_data.ranges_m[sample_index], active=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = FakeProximityCsvReplay()
        rclpy.spin(node)
    except (ValueError, OSError) as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f"Error: {error}")
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

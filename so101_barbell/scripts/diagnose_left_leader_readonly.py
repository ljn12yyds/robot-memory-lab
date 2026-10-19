#!/usr/bin/env python3
"""只读检查左主臂电压和状态寄存器，并把结果写入 JSON。

不启用扭矩、不写目标位置。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader


PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_LEADER_SERIAL-if00"
OUTPUT = Path("reports/left_leader_readonly_diagnostic.json")


def read_optional(bus, register: str, normalize: bool = False):
    """读取可选寄存器；固件不支持时记录异常，而不是中断整个诊断。"""
    try:
        return bus.sync_read(register, normalize=normalize)
    except Exception as error:
        return {"_error": repr(error)}


def main() -> None:
    """只读采集主臂状态并写入 JSON。"""
    # ---------- 1. 参数与主臂对象 ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--id", default="SO101_LEADER_ID")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--interval-s", type=float, default=0.1)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.interval_s < 0:
        parser.error("--interval-s must be >= 0")
    leader = SO101Leader(SO101LeaderConfig(port=args.port, id=args.id, use_degrees=True))
    # ---------- 2. 读取位置、电压、温度和错误状态 ----------
    leader.bus.connect()
    try:
        voltage_samples_raw = []
        voltage_read_errors = []
        for sample_index in range(args.samples):
            sample = read_optional(leader.bus, "Present_Voltage")
            if isinstance(sample, dict) and "_error" not in sample:
                voltage_samples_raw.append(sample)
            else:
                voltage_read_errors.append(
                    {"sample_index": sample_index, "error": sample.get("_error", str(sample))}
                )
            if sample_index + 1 < args.samples and args.interval_s:
                time.sleep(args.interval_s)

        voltage_stats = {}
        motors = list(leader.bus.motors)
        for motor in motors:
            values = [float(sample[motor]) for sample in voltage_samples_raw if motor in sample]
            if not values:
                continue
            voltage_stats[motor] = {
                "count": len(values),
                "raw_min": min(values),
                "raw_max": max(values),
                "raw_mean": statistics.fmean(values),
                "volts_min": min(values) / 10.0,
                "volts_max": max(values) / 10.0,
                "volts_mean": statistics.fmean(values) / 10.0,
            }

        result = {
            "port": args.port,
            "id": args.id,
            "voltage_sample_count_requested": args.samples,
            "voltage_sample_interval_s": args.interval_s,
            "voltage_samples_raw": voltage_samples_raw,
            "voltage_read_errors": voltage_read_errors,
            "voltage_stats": voltage_stats,
            "configured_min_voltage_raw": read_optional(leader.bus, "Min_Voltage_Limit"),
            "configured_max_voltage_raw": read_optional(leader.bus, "Max_Voltage_Limit"),
            "present_temperature_raw": read_optional(leader.bus, "Present_Temperature"),
            "present_load_raw": read_optional(leader.bus, "Present_Load"),
            "present_current_raw": read_optional(leader.bus, "Present_Current"),
            # Direct bus diagnostics intentionally do not register calibration;
            # raw encoder positions remain useful and avoid a misleading
            # "no calibration registered" normalization error.
            "present_position_raw": read_optional(
                leader.bus, "Present_Position", normalize=False
            ),
            "torque_enable": read_optional(leader.bus, "Torque_Enable"),
            "status": read_optional(leader.bus, "Status"),
            "hardware_error_status": read_optional(leader.bus, "Hardware_Error_Status"),
            "read_only": True,
            "motion_command_sent": False,
        }
    finally:
        leader.bus.disconnect(disable_torque=False)
    # ---------- 3. 保存诊断结果 ----------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("READ ONLY: no torque, calibration, or goal-position command was written.")


if __name__ == "__main__":
    main()

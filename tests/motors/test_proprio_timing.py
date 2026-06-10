#!/usr/bin/env python3
"""Timing validation for proprioceptive register reads on STS3215.

Tests whether adding sync_read("Present_Current") alongside
sync_read("Present_Position") stays within the 30 Hz budget (33ms).

Requires: SO-101 follower arm connected via USB.

Usage:
    python tests/motors/test_proprio_timing.py --port /dev/ttyACM0
    python tests/motors/test_proprio_timing.py --port /dev/ttyACM0 --cycles 200
"""
import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser(description="Timing test for dual sync_read")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--cycles", type=int, default=100)
    args = parser.parse_args()

    try:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
    except ImportError:
        print("ERROR: lerobot not installed. Run: uv pip install -e ./lerobot[feetech]")
        sys.exit(1)

    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }

    bus = FeetechMotorsBus(port=args.port, motors=motors)
    bus.connect()

    N = args.cycles
    print(f"Port: {args.port}")
    print(f"Motors: {len(motors)}")
    print(f"Cycles: {N}")
    print("=" * 60)

    # Test 1: Position only (baseline)
    print("\n[Test 1] Position only (baseline):")
    times_pos = []
    for _ in range(N):
        t0 = time.perf_counter()
        bus.sync_read("Present_Position", normalize=False)
        times_pos.append(time.perf_counter() - t0)

    mean_pos = sum(times_pos) / len(times_pos) * 1000
    max_pos = max(times_pos) * 1000
    hz_pos = 1000 / mean_pos
    print(f"  Mean: {mean_pos:.2f} ms | Max: {max_pos:.2f} ms | Rate: {hz_pos:.1f} Hz")

    # Test 2: Position + Current
    print("\n[Test 2] Position + Current:")
    times_dual = []
    for _ in range(N):
        t0 = time.perf_counter()
        bus.sync_read("Present_Position", normalize=False)
        bus.sync_read("Present_Current", normalize=False)
        times_dual.append(time.perf_counter() - t0)

    mean_dual = sum(times_dual) / len(times_dual) * 1000
    max_dual = max(times_dual) * 1000
    hz_dual = 1000 / mean_dual
    print(f"  Mean: {mean_dual:.2f} ms | Max: {max_dual:.2f} ms | Rate: {hz_dual:.1f} Hz")

    # Test 3: Position + Current + Temperature
    print("\n[Test 3] Position + Current + Temperature:")
    times_triple = []
    for _ in range(N):
        t0 = time.perf_counter()
        bus.sync_read("Present_Position", normalize=False)
        bus.sync_read("Present_Current", normalize=False)
        bus.sync_read("Present_Temperature", normalize=False)
        times_triple.append(time.perf_counter() - t0)

    mean_triple = sum(times_triple) / len(times_triple) * 1000
    max_triple = max(times_triple) * 1000
    hz_triple = 1000 / mean_triple
    print(f"  Mean: {mean_triple:.2f} ms | Max: {max_triple:.2f} ms | Rate: {hz_triple:.1f} Hz")

    # Test 4: Position + Current + Temperature + Voltage
    print("\n[Test 4] Position + Current + Temperature + Voltage:")
    times_all = []
    for _ in range(N):
        t0 = time.perf_counter()
        bus.sync_read("Present_Position", normalize=False)
        bus.sync_read("Present_Current", normalize=False)
        bus.sync_read("Present_Temperature", normalize=False)
        bus.sync_read("Present_Voltage", normalize=False)
        times_all.append(time.perf_counter() - t0)

    mean_all = sum(times_all) / len(times_all) * 1000
    max_all = max(times_all) * 1000
    hz_all = 1000 / mean_all
    print(f"  Mean: {mean_all:.2f} ms | Max: {max_all:.2f} ms | Rate: {hz_all:.1f} Hz")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<35} {'Mean ms':>8} {'Max ms':>8} {'Hz':>8} {'30Hz?':>6}")
    print("-" * 65)
    print(f"{'Position only':<35} {mean_pos:8.2f} {max_pos:8.2f} {hz_pos:8.1f} {'YES' if mean_pos < 33 else 'NO':>6}")
    print(f"{'Position + Current':<35} {mean_dual:8.2f} {max_dual:8.2f} {hz_dual:8.1f} {'YES' if mean_dual < 33 else 'NO':>6}")
    print(f"{'Position + Current + Temp':<35} {mean_triple:8.2f} {mean_triple:8.2f} {hz_triple:8.1f} {'YES' if mean_triple < 33 else 'NO':>6}")
    print(f"{'Position + Current + Temp + Volt':<35} {mean_all:8.2f} {max_all:8.2f} {hz_all:8.1f} {'YES' if mean_all < 33 else 'NO':>6}")

    # Decision
    print("\n" + "=" * 60)
    if mean_dual < 33:
        print("DECISION: Position + Current fits in 30 Hz budget.")
        print("          Safe to use log_current=True at 30 FPS.")
    elif mean_dual < 50:
        print("DECISION: Position + Current exceeds 30 Hz but fits 20 Hz.")
        print("          Recommend: set dataset.fps=20 or implement bulk read.")
    else:
        print("DECISION: Position + Current too slow even for 20 Hz.")
        print("          Need bulk/block read optimization.")
    print("=" * 60)

    bus.disconnect()


if __name__ == "__main__":
    main()

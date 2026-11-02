#!/usr/bin/env python3
"""HugWBC deployment-package integrity check (no Isaac Gym, no DDS, no robot).

Verifies both R41 and R42 packages against the frozen training contract:
file hashes, observation/action contract, the 23->29 SDK mapping with the six
held wrists, the trained leg/waist + accepted arm PD tables, arm reference
parameters and model identity. Field checks use PyYAML when available and are
skipped with a warning otherwise (hash checks still run).
"""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    "format": "hugwbc_g1_23_88_v1",
    "checkpoint_sha256": {
        "g1r41": "2e468ff18b5d34797530182c1e87f0e09441ee11aadd24661fff10c1042f6b51",
        "g1r42": "4933371459e0e26ec74833c4161be6774201c7158d6810334578e89bec37cb04",
    },
    "source_commit": {
        "g1r41": "b219b94b433d077fca673c945c22f67ab03ff3f3",
        "g1r42": "4c1234ba91a71143cf6f85eb2e33d139f15b36d0",
    },
    "sdk_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 22, 23, 24, 25],
    "wrist_sdk_indices": [19, 20, 21, 26, 27, 28],
    "joint_names": [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ],
    "default_joint_pos": [-0.1, 0, 0, 0.3, -0.2, 0] * 2 + [0, 0, 0] + [0] * 8,
    # G1InterruptCfg.control (leg/waist) + accepted G1Cfg arm override.
    "kp": [100, 100, 100, 150, 20, 20, 100, 100, 100, 150, 20, 20, 150, 150, 150,
           90, 60, 20, 60, 90, 60, 20, 60],
    "kd": [2.5, 2.5, 2.5, 3, 1, 1, 2.5, 2.5, 2.5, 3, 1, 1, 3, 3, 3,
           2, 1, 0.4, 1, 2, 1, 0.4, 1],
    "torque_limits": [88, 88, 88, 140, 45, 45, 88, 88, 88, 140, 45, 45, 88, 88, 88,
                      25, 25, 25, 25, 25, 25, 25, 25],
    "action_scale": [0.25] * 23,
    "command_scales": [2.0, 2.0, 0.25, 1.0, 1.0, 1.0, 0.15, 2.0, 0.5, 0.5, 1.0],
    "test_command": [0.5, 0.0, 0.0, 1.5, 0.5, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
    "arm_reference": {"model_indices": [15, 16, 17, 18, 19, 20, 21, 22],
                      "pitch_bias_rad": 0.3, "shoulder_roll_rad": 0.27, "elbow_rad": 0.97,
                      "swing_pitch_amplitude": 0.09178921571001411,
                      "swing_phase_offset": 0.30418359554562746,
                      "swing_speed_reference": 0.5, "velocity_feedforward": False},
    "obs_scales": {"ang_vel": 0.25, "dof_vel": 0.05},
    "dims": {"observation_dim": 88, "history_steps": 5, "action_dim": 23,
             "proprioception_dim": 75, "command_dim": 11, "clock_dim": 2},
    "step_dt": 0.02,
    "clip": {"clip_observations": 100.0, "clip_actions": 100.0},
    "torque_guard": True,
    "max_target_speed": 0.0,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tol=1e-9):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(y, str):
            if x != y:
                return False
        elif abs(float(x) - float(y)) > tol:
            return False
    return True


def check_package(package: Path, task: str):
    manifest = json.loads((package / "manifest.json").read_text())
    # Hash identity, independent of any yaml parser.
    if manifest["checkpoint_sha256"] != EXPECTED["checkpoint_sha256"][task]:
        raise SystemExit(f"FAILED: {package}: checkpoint is not the frozen {task} model")
    if manifest.get("source_commit") != EXPECTED["source_commit"][task]:
        raise SystemExit(f"FAILED: {package}: unexpected source commit")
    for name, key in (("policy.onnx", "policy_sha256"), ("deploy.yaml", "deploy_yaml_sha256")):
        if sha(package / name) != manifest[key]:
            raise SystemExit(f"FAILED: {package}: checksum mismatch for {name}")
    if (package / "policy.onnx").stat().st_size < 1000:
        raise SystemExit(f"FAILED: {package}: policy.onnx implausibly small")
    print(f"PASS: {package.name}: hashes and {task} model identity")

    try:
        import yaml
    except ImportError:
        print(f"WARN: {package.name}: PyYAML unavailable; deploy.yaml field checks skipped")
        return
    cfg = yaml.safe_load((package / "deploy.yaml").read_text())
    failures = []

    def expect(field, value):
        actual = cfg
        for part in field.split("."):
            actual = actual.get(part) if isinstance(actual, dict) else None
        if isinstance(value, list):
            ok = actual is not None and close(actual, value)
        elif isinstance(value, float):
            ok = actual is not None and abs(float(actual) - value) <= 1e-9
        else:
            ok = actual == value
        if not ok:
            failures.append(f"{field}: expected {value}, got {actual}")

    expect("format", EXPECTED["format"])
    for field, value in EXPECTED["dims"].items():
        expect(field, value)
    expect("step_dt", EXPECTED["step_dt"])
    expect("joint_names", EXPECTED["joint_names"])
    expect("sdk_indices", EXPECTED["sdk_indices"])
    expect("wrist_hold.sdk_indices", EXPECTED["wrist_sdk_indices"])
    expect("default_joint_pos", EXPECTED["default_joint_pos"])
    expect("kp", EXPECTED["kp"])
    expect("kd", EXPECTED["kd"])
    expect("torque_limits", EXPECTED["torque_limits"])
    expect("action_scale", EXPECTED["action_scale"])
    expect("command_scales", EXPECTED["command_scales"])
    for field, value in EXPECTED["clip"].items():
        expect(field, value)
    for field, value in EXPECTED["obs_scales"].items():
        expect(f"obs_scales.{field}", value)
    for field, value in EXPECTED["arm_reference"].items():
        expect(f"arm_reference.{field}", value)
    expect("torque_guard", EXPECTED["torque_guard"])
    expect("max_target_speed", EXPECTED["max_target_speed"])
    expect("checkpoint_sha256", EXPECTED["checkpoint_sha256"][task])
    expect("policy_sha256", manifest["policy_sha256"])
    if cfg.get("imu_source") not in ("pelvis", "torso"):
        failures.append(f"imu_source: unexpected value {cfg.get('imu_source')}")
    if failures:
        raise SystemExit(f"FAILED: {package}:\n  " + "\n  ".join(failures))
    print(f"PASS: {package.name}: contract fields (dims, 23->29 mapping, PD, arms, scales)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1] / "deploy/robots/g1_29dof/config/policy/hugwbc"
    parser.add_argument("package", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    packages = args.package or [(root / "r41", "g1r41"), (root / "r42", "g1r42")]
    for item in packages:
        package, task = item if isinstance(item, tuple) else (item, None)
        if task is None:
            task = package.name if package.name in EXPECTED["checkpoint_sha256"] else None
            if task is None:
                raise SystemExit(f"FAILED: cannot infer task variant from {package}")
        check_package(package.resolve(), task)


if __name__ == "__main__":
    main()

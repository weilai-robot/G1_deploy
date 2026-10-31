#!/usr/bin/env python3
"""Independent check of the 154-observation contract used by holo_wbt.

Two modes:

* package only: reconstruct an ideal robot state from the reference table and
  print the first frames of the expected observation, verifying the reference
  table, deploy.yaml and the ONNX interface.
* with --offline: run the compiled `holo_wbt_offline` executable on the same
  states and compare its 154 observations with this implementation element by
  element (tolerance 1e-5). This is the parity gate before touching hardware.

Usage:
    python tools/verify_holo_wbt_offline.py --package deploy/robots/g1_29dof/config/policy/holo_wbt/czy_box
    python tools/verify_holo_wbt_offline.py --package ... --offline build/holo_wbt_offline
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import yaml

OBS_DIM = 154
ACTION_DIM = 29
BODY_COUNT = 14
ANCHOR_INDEX = 7  # torso_link
TOLERANCE = 1e-5


def rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """Rz(r) @ Ry(p) @ Rx(y), matching the C++ forward kinematics."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def forward_kinematics(chain: list[dict], q: np.ndarray) -> np.ndarray:
    """3x3 rotation of the chain tip, translations do not affect orientation."""
    rotation = np.eye(3)
    for segment in chain:
        rotation = rotation @ rotation_from_rpy(np.asarray(segment["rpy"], dtype=np.float64))
        index = int(segment["motor_index"])
        if index >= 0:
            rotation = rotation @ axis_rotation(np.asarray(segment["axis"], dtype=np.float64), float(q[index]))
    return rotation


def matrix_to_wxyz(rotation: np.ndarray) -> list[float]:
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2.0
    x = np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0
    if m[2, 1] - m[1, 2] < 0:
        x = -x
    if m[0, 2] - m[2, 0] < 0:
        y = -y
    if m[1, 0] - m[0, 1] < 0:
        z = -z
    quat = np.array([w, x, y, z], dtype=np.float64)
    return (quat / np.linalg.norm(quat)).tolist()


def wxyz_to_matrix(quat: list[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_reference(path: Path) -> dict:
    with path.open("rb") as handle:
        magic, frames, joints, bodies, dt = struct.unpack("<8sIIIf", handle.read(24))
        if magic != b"BMG1V1\0\0" or joints != ACTION_DIM or bodies != BODY_COUNT:
            raise ValueError(f"unexpected reference header: {magic} {joints} {bodies}")
        records = np.frombuffer(handle.read(), dtype="<f4").reshape(frames, 2 * ACTION_DIM + 7 * BODY_COUNT)
    if path.stat().st_size != 24 + frames * records.shape[1] * 4:
        raise ValueError("reference table size mismatch")
    return {
        "frames": frames,
        "dt": dt,
        "q": records[:, :ACTION_DIM].astype(np.float64),
        "dq": records[:, ACTION_DIM:2 * ACTION_DIM].astype(np.float64),
        "pos": records[:, 2 * ACTION_DIM:2 * ACTION_DIM + 3 * BODY_COUNT].reshape(frames, BODY_COUNT, 3),
        "quat": records[:, 2 * ACTION_DIM + 3 * BODY_COUNT:].reshape(frames, BODY_COUNT, 4),
    }


def expected_observation(config: dict, reference: dict, frame: int, tilt_yaw: float = 0.0,
                         label: str = "") -> dict:
    """Build the state and the observation a correct implementation must produce."""
    chain = config["kinematics"]["torso"]
    q_policy = reference["q"][frame].copy()
    dq = reference["dq"][frame].copy()
    sdk_indices = np.asarray(config["sdk_indices"], dtype=int)
    q_sdk = np.zeros(ACTION_DIM)
    q_sdk[sdk_indices] = q_policy

    fk = forward_kinematics(chain, q_sdk)
    ref_torso = wxyz_to_matrix(reference["quat"][frame, ANCHOR_INDEX])
    yaw = np.array([
        [np.cos(tilt_yaw), -np.sin(tilt_yaw), 0.0],
        [np.sin(tilt_yaw), np.cos(tilt_yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    pelvis = yaw @ ref_torso @ fk.T
    torso = pelvis @ fk
    relative = torso.T @ ref_torso

    observation = np.concatenate([
        np.zeros(ACTION_DIM),                    # actions
        np.zeros(3),                             # base_ang_vel
        q_policy - np.asarray(config["default_joint_pos"], dtype=np.float64),  # dof_pos
        dq,                                      # dof_vel
        q_policy,                                # motion_command: joint_pos
        dq,                                      # motion_command: joint_vel
        relative[:, :2].reshape(-1),             # motion_ref_ori_b
    ])
    if observation.shape != (OBS_DIM,):
        raise ValueError(f"observation has {observation.shape[0]} entries, expected {OBS_DIM}")
    return {
        "label": label,
        "line": {
            "q": q_sdk.tolist(),
            "dq": dq.tolist(),
            "quat": matrix_to_wxyz(pelvis),
            "omega": [0.0, 0.0, 0.0],
            "frame": frame,
            "reset": True,
        },
        "expected": observation,
    }


def compare_with_binary(binary: Path, package: Path, states: list[dict]) -> float:
    stdin = "\n".join(json.dumps(state["line"]) for state in states) + "\n"
    completed = subprocess.run([str(binary), str(package)], input=stdin, text=True,
                               capture_output=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"holo_wbt_offline failed: {completed.stderr.strip()}")
    outputs = [json.loads(line) for line in completed.stdout.strip().splitlines()]
    if len(outputs) != len(states):
        raise SystemExit(f"expected {len(states)} replies, got {len(outputs)}")
    worst = 0.0
    for index, (state, reply) in enumerate(zip(states, outputs)):
        actual = np.asarray(reply["obs"], dtype=np.float64)
        expected = state["expected"]
        difference = np.abs(actual - expected)
        worst = max(worst, float(difference.max()))
        if difference.max() > TOLERANCE:
            bad = np.argsort(difference)[-5:][::-1]
            details = ", ".join(f"[{i}] c++={actual[i]:.6f} py={expected[i]:.6f}" for i in bad)
            raise SystemExit(f"frame {state['line']['frame']}: max |diff| {difference.max():.3e} > {TOLERANCE:g} at {details}")
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True, help="holo_wbt policy package directory")
    parser.add_argument("--offline", type=Path, default=None, help="compiled holo_wbt_offline executable")
    parser.add_argument("--tilt", type=float, default=0.2, help="pelvis yaw offset used for the non-trivial case")
    arguments = parser.parse_args()

    package = arguments.package.resolve()
    config = yaml.safe_load((package / "deploy.yaml").read_text(encoding="utf-8"))
    if config["format"] != "holo_wbt_g1_154_v1" or config["observation_dim"] != OBS_DIM:
        raise ValueError("package is not a holo_wbt_g1_154_v1 contract")
    if len(config["joint_names"]) != ACTION_DIM or len(config["sdk_indices"]) != ACTION_DIM:
        raise ValueError("package joint tables are not 29 wide")
    reference = load_reference(package / "reference.bin")
    if abs(reference["dt"] - 0.02) > 1e-6:
        raise ValueError(f"reference step is {reference['dt']}, expected 0.02")
    if not np.allclose(np.linalg.norm(reference["quat"], axis=-1), 1.0, atol=0.01):
        raise ValueError("reference quaternions are not normalised")

    import onnxruntime as ort
    session = ort.InferenceSession(str(package / "policy.onnx"), providers=["CPUExecutionProvider"])
    if [item.name for item in session.get_inputs()] != ["obs"] or list(session.get_inputs()[0].shape) != [1, OBS_DIM]:
        raise ValueError("policy.onnx does not take a single [1,154] observation")
    if [item.name for item in session.get_outputs()] != ["actions"] or list(session.get_outputs()[0].shape) != [1, ACTION_DIM]:
        raise ValueError("policy.onnx does not return a single [1,29] action")

    frames = sorted({0, reference["frames"] // 2, reference["frames"] - 1})
    states = [expected_observation(config, reference, frame, 0.0, "ideal") for frame in frames]
    states += [expected_observation(config, reference, frame, arguments.tilt, f"tilt{arguments.tilt:g}")
               for frame in frames]

    for state in states:
        action = session.run(["actions"], {"obs": state["expected"].astype(np.float32).reshape(1, OBS_DIM)})[0]
        if not np.isfinite(action).all():
            raise ValueError(f"non-finite action at frame {state['line']['frame']}")
        state["action"] = action.reshape(-1)

    print(f"package  {package}")
    print(f"frames   {reference['frames']} @ {1.0 / reference['dt']:.0f} Hz")
    print(f"checks   {len(states)} states (perfect tracking + {arguments.tilt:g} rad pelvis yaw offset)")
    for state in states:
        frame = state["line"]["frame"]
        obs = state["expected"]
        ori = obs[148:154]
        label = state["label"]
        print(f"frame {frame:3d} {label:>10s} ori={np.round(ori, 4).tolist()} |action|max={np.abs(state['action']).max():.3f}")

    if arguments.offline is None:
        print("no --offline executable: skipped C++ parity (run this on Linux after building)")
        return
    worst = compare_with_binary(arguments.offline, package, states)
    print(f"C++ parity passed on {len(states)} states, max |diff| = {worst:.3e}")


if __name__ == "__main__":
    main()

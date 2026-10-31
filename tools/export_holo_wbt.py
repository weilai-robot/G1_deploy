#!/usr/bin/env python3
"""Package a holosoma G1-29DOF whole-body-tracking PPO checkpoint for G1_deploy.

The holosoma WBT policy consumes 154 raw observations. The actor group is
concatenated in alphabetically sorted term-name order, which gives:

    actions          29  (previous policy action)
    base_ang_vel      3
    dof_pos          29  (joint position minus default)
    dof_vel          29
    motion_command   58  (reference joint position 29 + joint velocity 29)
    motion_ref_ori_b  6  (anchor-body orientation error, 3x3 matrix first two columns)

This is the BeyondMimic/whole_body_tracking observation minus motion_ref_pos_b (3)
and minus base_lin_vel (3), so no leg odometry is needed on the robot.

Outputs written to --output:

    policy.onnx     input "obs" [1,154] -> output "actions" [1,29], normalizer embedded
    reference.bin   BMG1V1 table (29 joints + 14 bodies, 50 Hz, wxyz quaternions)
    deploy.yaml     holo_wbt_g1_154_v1 control/observation contract
    manifest.json   checksums, frame count, onnx parity error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch import nn

OBS_DIM = 154
ACTION_DIM = 29
MAGIC = b"BMG1V1\0\0"
REF_FORMAT = "BMG1V1"
DEPLOY_FORMAT = "holo_wbt_g1_154_v1"

OBS_TERMS = [
    # holosoma concatenates an observation group in alphabetically sorted term order.
    ("actions", 29),
    ("base_ang_vel", 3),
    ("dof_pos", 29),
    ("dof_vel", 29),
    ("motion_command", 58),
    ("motion_ref_ori_b", 6),
]

# Unitree G1 29DOF hardware (DDS) joint order.
SDK_JOINT_ORDER = [
    *[f"{side}_{joint}_joint" for side in ("left", "right")
      for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")],
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *[f"{side}_{joint}_joint" for side in ("left", "right")
      for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                    "wrist_roll", "wrist_pitch", "wrist_yaw")],
]

# Bodies carried in the reference table; only torso_link (anchor) and the two
# ankle_roll links are consumed by the controller, the rest keep the format
# identical to the existing BeyondMimic packages.
BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]

BODY_TIPS = {
    "torso": "torso_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _activation_elu() -> nn.Module:
    return nn.ELU()


class Policy(nn.Module):
    """Actor MLP plus the empirical observation normalizer, as trained."""

    def __init__(self, layers: list[nn.Linear], mean: torch.Tensor, std: torch.Tensor, eps: float) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for index, layer in enumerate(layers):
            modules.append(layer)
            if index + 1 != len(layers):
                modules.append(_activation_elu())
        self.net = nn.Sequential(*modules)
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())
        self.eps = float(eps)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net((obs - self.mean) / (self.std + self.eps))


def build_policy(checkpoint: dict, eps: float) -> Policy:
    state = checkpoint.get("actor_model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no actor_model_state_dict; not a holosoma PPO checkpoint")
    pattern = re.compile(r"actor_module\.module\.(\d+)\.weight")
    ids = sorted(int(m.group(1)) for key in state if (m := pattern.fullmatch(key)))
    if not ids:
        raise ValueError("cannot find actor_module.module.* layers in the checkpoint")
    layers = []
    for index in ids:
        weight = state[f"actor_module.module.{index}.weight"]
        bias = state[f"actor_module.module.{index}.bias"]
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        layers.append(layer)
    if layers[0].in_features != OBS_DIM or layers[-1].out_features != ACTION_DIM:
        raise ValueError(
            f"actor is {layers[0].in_features} -> {layers[-1].out_features}, "
            f"expected {OBS_DIM} -> {ACTION_DIM}"
        )
    normalizer = checkpoint.get("actor_obs_normalizer_state_dict")
    if not isinstance(normalizer, dict):
        raise ValueError("checkpoint has no actor_obs_normalizer_state_dict (empirical normalization was disabled)")
    mean, std = normalizer["_mean"], normalizer["_std"]
    if mean.numel() != OBS_DIM or std.numel() != OBS_DIM:
        raise ValueError(f"normalizer is {mean.numel()} wide, expected {OBS_DIM}")
    for tensor, name in ((mean, "mean"), (std, "std")):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite normalizer {name}")
    if (std < 0).any():
        raise ValueError("negative normalizer std")
    for layer in layers:
        for parameter in layer.parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError("non-finite actor weights")
    return Policy(layers, mean, std, eps).eval()


def export_onnx(policy: Policy, path: Path, opset: int) -> None:
    options = dict(
        input_names=["obs"],
        output_names=["actions"],
        opset_version=opset,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(policy, torch.zeros(1, OBS_DIM), str(path), dynamo=False, **options)
        except TypeError:
            # torch < 2.1 has no `dynamo` switch; the tracer is the only exporter.
            torch.onnx.export(policy, torch.zeros(1, OBS_DIM), str(path), **options)
    try:
        import onnx  # optional: only used to stamp metadata
    except ImportError:
        print("WARNING: python package 'onnx' is unavailable; skipping onnx metadata")
        return
    proto = onnx.load(str(path))
    onnx.helper.set_model_props(proto, {"format": DEPLOY_FORMAT, "normalizer": "embedded", "observation_dim": str(OBS_DIM)})
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))


def check_onnx_parity(policy: Policy, path: Path, samples: int = 64, seed: int = 17) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = [item.name for item in session.get_inputs()]
    outputs = [item.name for item in session.get_outputs()]
    shape = session.get_inputs()[0].shape
    if inputs != ["obs"] or outputs != ["actions"] or list(shape) != [1, OBS_DIM]:
        raise ValueError(f"unexpected onnx interface: inputs={inputs} {shape} outputs={outputs}")
    generator = np.random.default_rng(seed)
    max_error = 0.0
    with torch.no_grad():
        for _ in range(samples):
            x = generator.normal(size=(1, OBS_DIM)).astype(np.float32)
            x = x * policy.std.numpy() + policy.mean.numpy()
            expected = policy(torch.from_numpy(x)).numpy()
            actual = session.run(["actions"], {"obs": x})[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return max_error


def load_reference(motion_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, list[str], list[str]]:
    with np.load(motion_path, allow_pickle=False) as data:
        fps = float(np.asarray(data["fps"]).flat[0])
        joint_names = [str(name) for name in data["joint_names"].tolist()]
        body_names = [str(name) for name in data["body_names"].tolist()]
        joint_pos, joint_vel = data["joint_pos"], data["joint_vel"]
        body_pos, body_quat = data["body_pos_w"], data["body_quat_w"]
    # holosoma layout: joint_pos starts with root xyz + wxyz quaternion (7 columns),
    # joint_vel starts with root linear + angular velocity (6 columns).
    if joint_pos.shape[1] != len(joint_names) + 7 or joint_vel.shape[1] != len(joint_names) + 6:
        raise ValueError(
            f"unexpected motion layout: joint_pos {joint_pos.shape}, joint_vel {joint_vel.shape}, "
            f"{len(joint_names)} joint names"
        )
    q = np.asarray(joint_pos[:, 7:], dtype=np.float64)
    dq = np.asarray(joint_vel[:, 6:], dtype=np.float64)
    if q.shape[1] != ACTION_DIM or dq.shape != q.shape:
        raise ValueError(f"reference has {q.shape[1]} robot joints, expected {ACTION_DIM}")
    if abs(fps - 50.0) > 1e-4:
        raise ValueError(f"reference motion is {fps} Hz, this contract requires 50 Hz")
    indexes = []
    for name in BODY_NAMES:
        if name not in body_names:
            raise ValueError(f"motion is missing body '{name}'")
        indexes.append(body_names.index(name))
    pos = np.asarray(body_pos[:, indexes], dtype=np.float64)
    quat = np.asarray(body_quat[:, indexes], dtype=np.float64)
    norms = np.linalg.norm(quat, axis=-1)
    if not np.allclose(norms, 1.0, atol=0.01):
        raise ValueError("reference quaternions are not normalised")
    for array, label in ((q, "joint_pos"), (dq, "joint_vel"), (pos, "body_pos"), (quat, "body_quat")):
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite reference {label}")
    return q, dq, pos, quat, fps, joint_names, body_names


def write_reference(path: Path, q: np.ndarray, dq: np.ndarray, pos: np.ndarray, quat: np.ndarray, fps: float) -> int:
    frames = len(q)
    records = np.concatenate([q, dq, pos.reshape(frames, -1), quat.reshape(frames, -1)], axis=1).astype("<f4")
    if records.shape[1] != 2 * ACTION_DIM + 7 * len(BODY_NAMES):
        raise ValueError(f"reference record has {records.shape[1]} floats")
    with path.open("wb") as handle:
        handle.write(struct.pack("<8sIIIf", MAGIC, frames, ACTION_DIM, len(BODY_NAMES), 1.0 / fps))
        handle.write(records.tobytes())
    if path.stat().st_size != 24 + frames * records.shape[1] * 4:
        raise ValueError("reference table size mismatch")
    return frames


def resolve_urdf(checkpoint_config: dict, holosoma_root: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    asset = checkpoint_config["robot"]["asset"]
    urdf_file = str(asset["urdf_file"])
    root = str(asset.get("asset_root", "@holosoma/data/robots"))
    root = root.replace("@holosoma", str(holosoma_root / "src" / "holosoma" / "holosoma"))
    return Path(root) / urdf_file


def build_kinematics(urdf_path: Path, joint_names: list[str]) -> dict[str, list[dict]]:
    root = ET.parse(urdf_path).getroot()
    links = [link.get("name") for link in root.findall("link")]
    by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
    roots = [name for name in links if name not in by_child]
    if roots != ["pelvis"]:
        raise ValueError(f"unexpected urdf root links: {roots}")

    def vector(element: ET.Element | None, attribute: str, fallback: str) -> list[float]:
        text = element.get(attribute, fallback) if element is not None else fallback
        values = [float(value) for value in text.split()]
        if len(values) != 3:
            raise ValueError(f"invalid urdf vector: {text}")
        return values

    def chain(tip: str) -> list[dict]:
        segments = []
        while tip != "pelvis":
            joint = by_child[tip]
            name = joint.get("name")
            segments.append({
                "motor_index": joint_names.index(name) if joint.get("type") != "fixed" else -1,
                "xyz": vector(joint.find("origin"), "xyz", "0 0 0"),
                "rpy": vector(joint.find("origin"), "rpy", "0 0 0"),
                "axis": vector(joint.find("axis"), "xyz", "0 0 1"),
            })
            tip = joint.find("parent").get("link")
        return list(reversed(segments))

    return {label: chain(tip) for label, tip in BODY_TIPS.items()}


def match_gain(table: dict, joint_name: str) -> float:
    matches = [float(value) for pattern, value in table.items() if pattern in joint_name]
    if len(matches) != 1:
        raise ValueError(f"expected one gain pattern for '{joint_name}', got {len(matches)}")
    return matches[0]


def build_deploy_config(checkpoint_config: dict, joint_names: list[str], kinematics: dict, motion_path: Path,
                        frames: int, fps: float, eps: float) -> dict:
    robot = checkpoint_config["robot"]
    control = robot["control"]
    if not control.get("action_scales_by_effort_limit_over_p_gain"):
        raise ValueError("checkpoint does not scale actions by effort_limit / p_gain")
    base_scale = float(control["action_scale"])
    defaults = robot["init_state"]["default_joint_angles"]
    effort = [float(value) for value in robot["dof_effort_limit_list"]]
    lower = [float(value) for value in robot["dof_pos_lower_limit_list"]]
    upper = [float(value) for value in robot["dof_pos_upper_limit_list"]]
    if not (len(effort) == len(lower) == len(upper) == ACTION_DIM):
        raise ValueError("robot limits/effort lists are not 29 wide")
    kp = [match_gain(control["stiffness"], name) for name in joint_names]
    kd = [match_gain(control["damping"], name) for name in joint_names]
    default = [float(defaults[name]) for name in joint_names]
    action_scale = [base_scale * effort[index] / kp[index] for index in range(ACTION_DIM)]
    for index, name in enumerate(joint_names):
        if not lower[index] < default[index] < upper[index]:
            raise ValueError(f"default position of '{name}' is outside its limits")
    return {
        "format": DEPLOY_FORMAT,
        "model": "policy.onnx",
        "reference": "reference.bin",
        "step_dt": 1.0 / fps,
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "obs_terms": [f"{name}:{dim}" for name, dim in OBS_TERMS],
        "joint_names": joint_names,
        "sdk_indices": [SDK_JOINT_ORDER.index(name) for name in joint_names],
        "body_names": BODY_NAMES,
        "anchor_body_name": "torso_link",
        "default_joint_pos": default,
        "kp": kp,
        "kd": kd,
        "action_scale": action_scale,
        "effort_limit": effort,
        "joint_lower": lower,
        "joint_upper": upper,
        "kinematics": kinematics,
        "normalizer": {"embedded": True, "epsilon": eps},
        "reference_frames": frames,
    }


def dump_yaml(data: dict, path: Path) -> None:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise SystemExit("PyYAML is required: pip install pyyaml") from error
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="holosoma model_XXXXX.pt")
    parser.add_argument("--motion", type=Path, required=True, help="holosoma motion npz used for training")
    parser.add_argument("--output", type=Path, required=True, help="policy package directory")
    parser.add_argument("--holosoma-root", type=Path, default=Path("e:/G1/holosoma"),
                        help="holosoma checkout, used to locate the training URDF")
    parser.add_argument("--urdf", type=Path, default=None, help="override the URDF used for kinematics")
    parser.add_argument("--normalizer-eps", type=float, default=0.01,
                        help="EmpiricalNormalization epsilon used during training")
    parser.add_argument("--opset", type=int, default=17)
    arguments = parser.parse_args()

    if not np.isfinite(arguments.normalizer_eps) or arguments.normalizer_eps <= 0:
        parser.error("--normalizer-eps must be positive and finite")

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "experiment_config" not in checkpoint:
        raise ValueError("checkpoint does not embed experiment_config; export from a holosoma PPO run")
    config = checkpoint["experiment_config"]
    joint_names = [str(name) for name in config["robot"]["dof_names"]]
    if len(joint_names) != ACTION_DIM or sorted(joint_names) != sorted(SDK_JOINT_ORDER):
        raise ValueError("robot joint names are not the Unitree G1 29DOF set")

    policy = build_policy(checkpoint, arguments.normalizer_eps)
    q, dq, pos, quat, fps, motion_joints, _ = load_reference(arguments.motion)
    if motion_joints != joint_names:
        raise ValueError("motion joint order differs from the checkpoint robot joint order")

    urdf = resolve_urdf(config, arguments.holosoma_root, arguments.urdf)
    if not urdf.exists():
        raise ValueError(f"URDF not found: {urdf}")
    kinematics = build_kinematics(urdf, joint_names)

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "policy.onnx"
    reference_path = output / "reference.bin"

    export_onnx(policy, onnx_path, arguments.opset)
    max_error = check_onnx_parity(policy, onnx_path)
    frames = write_reference(reference_path, q, dq, pos, quat, fps)
    deploy = build_deploy_config(config, joint_names, kinematics, arguments.motion, frames, fps,
                                 arguments.normalizer_eps)
    deploy_path = output / "deploy.yaml"
    dump_yaml(deploy, deploy_path)

    manifest = {
        "checkpoint": arguments.checkpoint.name,
        "checkpoint_sha256": sha256(arguments.checkpoint),
        "iteration": int(checkpoint.get("iter", -1)),
        "motion": arguments.motion.name,
        "motion_sha256": sha256(arguments.motion),
        "urdf": str(urdf),
        "urdf_sha256": sha256(urdf),
        "policy_sha256": sha256(onnx_path),
        "reference_sha256": sha256(reference_path),
        "deploy_yaml_sha256": sha256(deploy_path),
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "frames": frames,
        "fps": fps,
        "normalizer_epsilon": arguments.normalizer_eps,
        "onnx_parity_samples": 64,
        "onnx_max_abs_error": max_error,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"policy.onnx    {onnx_path}  ({OBS_DIM} -> {ACTION_DIM})")
    print(f"reference.bin  {reference_path}  ({frames} frames @ {fps:g} Hz)")
    print(f"deploy.yaml    {deploy_path}  ({DEPLOY_FORMAT})")
    print(f"manifest.json  {output / 'manifest.json'}")
    print(f"onnx max abs error vs torch: {max_error:.3e}")


if __name__ == "__main__":
    main()

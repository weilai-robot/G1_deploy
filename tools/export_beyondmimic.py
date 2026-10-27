#!/usr/bin/env python3
"""Package this repository's 160-observation G1 BeyondMimic checkpoint without Isaac Lab.

The actor ONNX takes RAW observations; the checkpoint's empirical normalization is
inside the graph. Reference motion is a separate versioned little-endian table.
"""
import argparse
import ast
import hashlib
import json
import operator
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import yaml

SDK_NAMES = [
    *[f"{s}_{j}_joint" for s in ("left", "right") for j in
      ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")],
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *[f"{s}_{j}_joint" for s in ("left", "right") for j in
      ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")],
]
BODY_NAMES = ["pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
              "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
              "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
              "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"]


def assignment(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise ValueError(f"Missing training definition: {name}")


def evaluate(node, constants):
    """Read numeric training settings without importing or executing training code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [evaluate(n, constants) for n in node.elts]
    if isinstance(node, ast.Dict):
        return {evaluate(k, constants): evaluate(v, constants) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -evaluate(node.operand, constants)
    if isinstance(node, ast.BinOp):
        op = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}[type(node.op)]
        return op(evaluate(node.left, constants), evaluate(node.right, constants))
    raise ValueError(f"Unsupported training expression: {ast.dump(node)}")


def kwargs(call):
    return {k.arg: k.value for k in call.keywords}


def training_parameters(path, names):
    tree = ast.parse(path.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = evaluate(node.value, constants)
            except (KeyError, ValueError, TypeError):
                pass
    robot = kwargs(assignment(tree, "G1_CYLINDER_CFG"))
    default = evaluate(kwargs(robot["init_state"])["joint_pos"], constants)
    actuator_cfg = robot["actuators"]
    actuators = [kwargs(v) for v in actuator_cfg.values]
    result = {key: [] for key in ("default_joint_pos", "kp", "kd", "action_scale", "effort_limit")}
    for name in names:
        matches = [a for a in actuators if any(re.fullmatch(p, name) for p in evaluate(a["joint_names_expr"], constants))]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous actuator for {name}")
        values = {}
        for key in ("stiffness", "damping", "effort_limit_sim"):
            value = evaluate(matches[0][key], constants)
            if isinstance(value, dict):
                candidates = [v for p, v in value.items() if re.fullmatch(p, name)]
                if len(candidates) != 1:
                    raise ValueError(f"Ambiguous {key} for {name}")
                value = candidates[0]
            values[key] = float(value)
        result["default_joint_pos"].append(next((float(v) for p, v in default.items() if re.fullmatch(p, name)), 0.0))
        result["kp"].append(values["stiffness"])
        result["kd"].append(values["damping"])
        result["effort_limit"].append(values["effort_limit_sim"])
        result["action_scale"].append(0.25 * values["effort_limit_sim"] / values["stiffness"])
    return result


class Actor(nn.Module):
    def __init__(self, checkpoint, eps):
        super().__init__()
        state = checkpoint.get("model_state_dict", checkpoint.get("model_dict"))
        if state is None:
            raise ValueError("Not an RSL-RL checkpoint")
        ids = sorted(int(k.split(".")[1]) for k in state if re.fullmatch(r"actor\.\d+\.weight", k))
        layers = []
        for i, idx in enumerate(ids):
            w, b = state[f"actor.{idx}.weight"], state[f"actor.{idx}.bias"]
            layer = nn.Linear(w.shape[1], w.shape[0])
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)
            layers.append(layer)
            if i + 1 != len(ids):
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)
        if layers[0].in_features != 160 or layers[-1].out_features != 29:
            raise ValueError("This integration requires the trained 160 -> 29 MLP")
        norm = checkpoint["obs_norm_state_dict"]
        self.register_buffer("mean", norm["_mean"].float())
        self.register_buffer("std", norm["_std"].float())
        self.eps = eps
        if self.mean.numel() != 160 or self.std.numel() != 160:
            raise ValueError("Normalizer dimensions do not match actor")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std < 0).any():
            raise ValueError("Invalid normalizer values")
        if any(not torch.isfinite(parameter).all() for parameter in self.actor.parameters()):
            raise ValueError("Non-finite actor weights")

    def forward(self, obs):
        return self.actor((obs - self.mean) / (self.std + self.eps))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--normalizer-eps", type=float, default=0.01,
                   help="RSL-RL EmpiricalNormalization default; must match original training")
    args = p.parse_args()
    if not np.isfinite(args.normalizer_eps) or args.normalizer_eps <= 0:
        p.error("--normalizer-eps must be positive and finite")
    train = args.training_root.resolve() / "whole_body_tracking"
    source = train / "source/whole_body_tracking/whole_body_tracking"
    agent_tree = ast.parse((source / "tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py").read_text())
    policy_defs = [n for n in ast.walk(agent_tree) if isinstance(n, ast.Call) and
                   isinstance(n.func, ast.Name) and n.func.id == "RslRlPpoActorCriticCfg"]
    if len(policy_defs) != 1 or ast.literal_eval(kwargs(policy_defs[0])["activation"]) != "elu":
        raise ValueError("This exporter requires the source task's ELU actor")
    playback = ast.parse((train / "scripts/play_mujoco.py").read_text())
    names = ast.literal_eval(assignment(playback, "ISAAC_JOINT_ORDER"))
    body_order = ast.literal_eval(assignment(playback, "NPZ_BODY_ORDER"))
    if len(names) != 29 or set(names) != set(SDK_NAMES):
        raise ValueError("Unsupported G1 joint names/order")
    params = training_parameters(source / "robots/g1.py", names)
    urdf = source / "assets/unitree_description/urdf/g1/main.urdf"
    # Extract exact kinematics/limits from the training robot, not a different G1 asset.
    root = ET.parse(urdf).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    by_name = {j.get("name"): j for j in root.findall("joint")}
    chains = {}
    for label, tip in (("torso", "torso_link"), ("left_foot", "left_ankle_roll_link"), ("right_foot", "right_ankle_roll_link")):
        chain = []
        while tip != "pelvis":
            j = by_child[tip]
            origin, axis = j.find("origin"), j.find("axis")
            get_vec = lambda element, attr, default: [float(x) for x in (element.get(attr, default) if element is not None else default).split()]
            chain.append({"motor_index": SDK_NAMES.index(j.get("name")) if j.get("type") != "fixed" else -1,
                          "xyz": get_vec(origin, "xyz", "0 0 0"), "rpy": get_vec(origin, "rpy", "0 0 0"),
                          "axis": get_vec(axis, "xyz", "0 0 1")})
            tip = j.find("parent").get("link")
        chains[label] = list(reversed(chain))
    for key in ("lower", "upper"):
        params["joint_" + key] = [float(by_name[n].find("limit").get(key)) for n in names]
    with np.load(args.motion, allow_pickle=False) as motion:
        fps = float(np.asarray(motion["fps"]).flat[0])
        if abs(fps - 50) > 1e-4:
            raise ValueError("This trained task requires a 50 Hz reference motion")
        indices = [body_order.index(n) for n in BODY_NAMES]
        q, dq = motion["joint_pos"], motion["joint_vel"]
        pos, quat = motion["body_pos_w"][:, indices], motion["body_quat_w"][:, indices]
        if q.ndim != 2 or q.shape[1] != 29 or dq.shape != q.shape or len(q) < 2:
            raise ValueError("Invalid reference joint arrays")
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=0.01):
            raise ValueError("Reference quaternion norm mismatch")
        records = np.concatenate([q, dq, pos.reshape(len(q), -1), quat.reshape(len(q), -1)], axis=1).astype("<f4")
    if not np.isfinite(records).all():
        raise ValueError("Non-finite reference data")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    actor = Actor(checkpoint, args.normalizer_eps).eval()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = out / "policy.onnx"
    torch.onnx.export(actor, torch.zeros(1, 160), str(model), input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    proto = onnx.load(str(model))
    onnx.helper.set_model_props(proto, {"format": "beyondmimic_g1_160_v1", "normalizer": "embedded",
                                       "joint_names": ",".join(names), "source_checkpoint_sha256": sha(args.checkpoint)})
    onnx.checker.check_model(proto)
    onnx.save(proto, str(model))
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(17)
    max_error = 0.0
    for _ in range(64):
        x = generator.normal(size=(1, 160)).astype(np.float32)
        # Sample near the training distribution, as well as unnormalised random inputs.
        x = x * actor.std.numpy() + actor.mean.numpy()
        expected = actor(torch.from_numpy(x)).detach().numpy()
        actual = session.run(["actions"], {"obs": x})[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    reference = out / "reference.bin"
    with reference.open("wb") as f:
        f.write(struct.pack("<8sIIIf", b"BMG1V1\0\0", len(q), 29, len(BODY_NAMES), 1 / fps))
        f.write(records.tobytes())
    config = {"format": "beyondmimic_g1_160_v1", "model": "policy.onnx", "reference": "reference.bin",
              "step_dt": 1 / fps, "observation_dim": 160, "action_dim": 29,
              "joint_names": names, "sdk_indices": [SDK_NAMES.index(n) for n in names],
              "body_names": BODY_NAMES, "anchor_body_name": "torso_link", **params, "kinematics": chains,
              "normalizer": {"embedded": True, "epsilon": args.normalizer_eps}}
    (out / "deploy.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {"checkpoint": args.checkpoint.name, "checkpoint_sha256": sha(args.checkpoint),
                "motion": args.motion.name, "motion_sha256": sha(args.motion),
                "policy_sha256": sha(model), "reference_sha256": sha(reference), "urdf_sha256": sha(urdf),
                "deploy_yaml_sha256": sha(out / "deploy.yaml"),
                "frames": len(q), "fps": fps, "normalizer_epsilon": args.normalizer_eps,
                "onnx_parity_samples": 64, "onnx_max_abs_error": max_error}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

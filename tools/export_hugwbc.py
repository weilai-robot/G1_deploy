#!/usr/bin/env python3
"""Export HugWBC R41/R42 actor checkpoints to G1 deployment packages (no Isaac Gym).

The checkpoint actor is the training ``MlpAdaptModel`` (external history, no
recurrence): ``mem_encoder`` + ``state_estimator`` + ``low_level_net``. The
exported ONNX replays exactly ``BaseAdaptModel.forward`` with
``privileged_obs=None`` and ``sync_update=False``:

    pro_history  = obs_history[..., :75]              # [1,5,75], old -> new
    latent       = mem_encoder(pro_history.flatten(-2,-1))   # [1,32]
    estimated    = state_estimator(latent)            # [1,3]
    action       = low_level_net(cat([latent, estimated,
                                      obs_history[..., -1, :75],
                                      obs_history[..., -1, 75:88]]))  # [1,23]

It is a plain deterministic mean action: no Gaussian sampling, no ``std``, no
critic/privileged ground truth, no observation normalizer (the HugWBC scaling is
applied explicitly by the deployment core, see deploy.yaml).

Contract values below are copied from the frozen training sources, not estimated:
- joint order / q0 / PD / torque limits: legged_gym/envs/g1/g1_config.py (G1Cfg)
  and legged_gym/envs/g1/g1interrupt_config.py (G1InterruptCfg.control), arm PD
  overridden by the accepted-manifest / g1r42_config.py G1Cfg arm values;
- observation scales and clips: legged_gym/envs/base/legged_robot_config.py
  (normalization.obs_scales / clip_observations / clip_actions);
- command scales: legged_gym/envs/h1/h1interrupt.py command_scale reconstruction;
- arm reference parameters: accepted/r41-reference-arms-20261030/manifest.json
  and controller-snapshot/g1r42.py (arm_rest_* / arm_swing_* from g1r29_config);
- position limits: x1-training/resources/robots/g1/g1_29dof_anneal_23dof.urdf.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import yaml

# --- frozen training contract -------------------------------------------------
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
]
# SDK motor index per model joint (g1_29dof SDK order; wrists 19-21/26-28 excluded).
SDK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25]
SDK_NAMES = [
    *[f"{s}_{j}_joint" for s in ("left", "right") for j in
      ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")],
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *[f"{s}_{j}_joint" for s in ("left", "right") for j in
      ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
       "wrist_roll", "wrist_pitch", "wrist_yaw")],
]
WRIST_SDK_INDICES = [19, 20, 21, 26, 27, 28]
ARM_MODEL_INDICES = [15, 16, 17, 18, 19, 20, 21, 22]
DEFAULT_Q = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0, 0.0, 0.0] + [0.0] * 8
# G1InterruptCfg.control (leg/waist) + accepted G1Cfg arm override, both versions.
KP = [100, 100, 100, 150, 20, 20, 100, 100, 100, 150, 20, 20, 150, 150, 150,
      90, 60, 20, 60, 90, 60, 20, 60]
KD = [2.5, 2.5, 2.5, 3.0, 1.0, 1.0, 2.5, 2.5, 2.5, 3.0, 1.0, 1.0, 3.0, 3.0, 3.0,
      2.0, 1.0, 0.4, 1.0, 2.0, 1.0, 0.4, 1.0]
# G1Cfg.control.torque_limits (simulation saturation values, not hardware ratings).
TORQUE_LIMITS = [88, 88, 88, 140, 45, 45, 88, 88, 88, 140, 45, 45, 88, 88, 88,
                 25, 25, 25, 25, 25, 25, 25, 25]
# g1_29dof_anneal_23dof.urdf revolute limits for the 23 policy joints.
JOINT_LOWER = [-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
               -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
               -2.618, -0.52, -0.52,
               -3.0892, -1.5882, -2.618, -1.0472,
               -3.0892, -2.2515, -2.618, -1.0472]
JOINT_UPPER = [2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
               2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
               2.618, 0.52, 0.52,
               2.6704, 2.2515, 2.618, 2.0944,
               2.6704, 1.5882, 2.618, 2.0944]
OBS_SCALES = {"ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}
COMMAND_SCALES = [2.0, 2.0, 0.25, 1.0, 1.0, 1.0, 0.15, 2.0, 0.5, 0.5, 1.0]
CLIP_OBSERVATIONS = 100.0
CLIP_ACTIONS = 100.0
ACTION_SCALE = 0.25
STEP_DT = 0.02  # sim.dt 0.005 x decimation 4
ARM_REFERENCE = {
    "pitch_bias_rad": 0.3,
    "shoulder_roll_rad": 0.27,
    "elbow_rad": 0.97,
    "swing_pitch_amplitude": 0.09178921571001411,
    "swing_phase_offset": 0.30418359554562746,
    "swing_speed_reference": 0.5,
    "velocity_feedforward": False,
}
# Fixed forward-walk evaluation command (eval_upright.py 'upright' protocol +
# baseline-hybrid/seed1/rollout.csv cmd_0..cmd_10); interrupt flag stays 0.
TEST_COMMAND = [0.5, 0.0, 0.0, 1.5, 0.5, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0]
WRIST_HOLD = {
    "sdk_indices": WRIST_SDK_INDICES,
    "target_rad": [0.0] * 6,
    # Deployment hold gains copied from this branch's FixStand wrist rows
    # (config.yaml); they are NOT trained HugWBC parameters.
    "kp": 40.0,
    "kd": 10.0,
}
FORMAT = "hugwbc_g1_23_88_v1"

SOURCE_COMMITS = {
    "g1r41": "b219b94b433d077fca673c945c22f67ab03ff3f3",
    "g1r42": "4c1234ba91a71143cf6f85eb2e33d139f15b36d0",
}

# --- actor rebuild ------------------------------------------------------------


def mlp(in_dim, out_dim, hidden, prefix, state):
    layers = []
    dims = [in_dim, *hidden, out_dim]
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        w = state[f"{prefix}.{2 * i}.weight"]
        b = state[f"{prefix}.{2 * i}.bias"]
        if tuple(w.shape) != (dims[i + 1], dims[i]) or tuple(b.shape) != (dims[i + 1],):
            raise ValueError(f"Unexpected {prefix}.{2*i} shape {tuple(w.shape)}")
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(b)
        layers.append(linear)
        if i + 2 != len(dims):
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class HugWBCActor(nn.Module):
    """Deterministic replay of training BaseAdaptModel.forward (MlpAdaptModel)."""

    def __init__(self, checkpoint):
        super().__init__()
        state = checkpoint.get("model_state_dict")
        if state is None:
            raise ValueError("Not an RSL-RL checkpoint (missing model_state_dict)")
        expect = {
            "mem_encoder": (375, [256, 128], 32),
            "state_estimator": (32, [64, 32], 3),
            "low_level_net": (123, [256, 128, 32], 23),
        }
        for name, (in_dim, hidden, out_dim) in expect.items():
            setattr(self, name, mlp(in_dim, out_dim, hidden, f"actor.{name}", state))
        self.proprioception_dim = 75
        self.cmd_dim = 13
        if any(not torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError("Non-finite actor weights")
        unexpected = {k for k in state if k.startswith("actor.") and
                      not k.startswith(("actor.mem_encoder.", "actor.state_estimator.",
                                        "actor.low_level_net."))}
        if unexpected:
            raise ValueError(f"Unexpected actor submodules in checkpoint: {sorted(unexpected)}")

    def forward(self, obs_history):
        pro_history = obs_history[..., : self.proprioception_dim]
        latent = self.mem_encoder(pro_history.flatten(-2, -1))
        estimated_priv = self.state_estimator(latent)
        latest_pro = obs_history[..., -1, : self.proprioception_dim]
        latest_cmd_clock = obs_history[..., -1, self.proprioception_dim:]
        return self.low_level_net(
            torch.cat((latent, estimated_priv, latest_pro, latest_cmd_clock), dim=-1))


def functional_actor(state, history):
    """Independent numpy/torch functional forward used to cross-check the rebuild."""
    def linear(x, prefix, idx):
        return torch.nn.functional.linear(
            x, state[f"actor.{prefix}.{idx}.weight"], state[f"actor.{prefix}.{idx}.bias"])

    def elu(x):
        return torch.nn.functional.elu(x)

    pro = history[..., :75]
    x = pro.reshape(pro.shape[0], -1)
    x = elu(linear(x, "mem_encoder", 0))
    x = elu(linear(x, "mem_encoder", 2))
    latent = linear(x, "mem_encoder", 4)
    e = elu(linear(latent, "state_estimator", 0))
    e = elu(linear(e, "state_estimator", 2))
    priv = linear(e, "state_estimator", 4)
    cat = torch.cat((latent, priv, history[:, -1, :75], history[:, -1, 75:88]), dim=-1)
    a = elu(linear(cat, "low_level_net", 0))
    a = elu(linear(a, "low_level_net", 2))
    a = elu(linear(a, "low_level_net", 4))
    return linear(a, "low_level_net", 6)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", choices=("g1r41", "g1r42"), required=True,
                   help="Training task identity recorded in the manifest")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parity-samples", type=int, default=128)
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    actor = HugWBCActor(checkpoint).eval()
    iteration = checkpoint.get("iter")

    # 1) rebuild vs independent functional forward on random histories
    generator = np.random.default_rng(20261102)
    for _ in range(16):
        history = torch.from_numpy(
            (generator.random((1, 5, 88)) * 2 - 1).astype(np.float32))
        rebuilt = actor(history)
        reference = functional_actor(checkpoint["model_state_dict"], history)
        torch.testing.assert_close(rebuilt, reference, rtol=1e-5, atol=1e-6)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "policy.onnx"
    torch.onnx.export(
        actor, torch.zeros(1, 5, 88), str(model_path),
        input_names=["obs_history"], output_names=["actions"],
        opset_version=17, dynamo=False)
    proto = onnx.load(str(model_path))
    onnx.helper.set_model_props(proto, {
        "format": FORMAT,
        "joint_names": ",".join(JOINT_NAMES),
        "source_checkpoint_sha256": sha(args.checkpoint),
    })
    onnx.checker.check_model(proto)
    onnx.save(proto, str(model_path))

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 5, 88] or session.get_outputs()[0].shape != [1, 23]:
        raise ValueError(f"Unexpected ONNX IO shapes {session.get_inputs()[0].shape} "
                         f"-> {session.get_outputs()[0].shape}")

    # 2) ONNX parity: near-realistic magnitudes plus unnormalised random inputs.
    #    Gate tolerance follows this repo's exporter convention (rtol/atol 2e-4,
    #    same as export_beyondmimic.py); the measured maxima are frozen in the
    #    manifest, not chosen to pass.
    max_error = 0.0
    realistic_error = 0.0
    for i in range(args.parity_samples):
        bucket = i % 4
        if bucket == 0:  # moderate random
            history = generator.uniform(-1.0, 1.0, (1, 5, 88))
        elif bucket == 1:  # realistic proprioception/commands, zero arm commands
            history = np.zeros((1, 5, 88), np.float32)
            history[..., :3] = generator.uniform(-0.5, 0.5, (1, 5, 3))
            history[..., 3:6] = generator.uniform(-1.0, 1.0, (1, 5, 3))
            history[..., 6:29] = generator.uniform(-0.6, 0.6, (1, 5, 23))
            history[..., 29:52] = generator.uniform(-1.5, 1.5, (1, 5, 23))
            history[..., 52:75] = generator.uniform(-1.0, 1.0, (1, 5, 23))
            history[..., 75:86] = np.array(COMMAND_SCALES, np.float32) * TEST_COMMAND
            history[..., 86] = generator.uniform(-1.0, 1.0)
            history[..., 87] = generator.uniform(-1.0, 1.0)
        else:
            history = generator.uniform(-100.0, 100.0, (1, 5, 88))
        history = history.astype(np.float32)
        expected = actor(torch.from_numpy(history)).detach().numpy()
        actual = session.run(["actions"], {"obs_history": history})[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
        error = float(np.max(np.abs(actual - expected)))
        max_error = max(max_error, error)
        if bucket == 1:
            realistic_error = max(realistic_error, error)

    deploy_yaml = {
        "format": FORMAT,
        "model": "policy.onnx",
        "checkpoint_sha256": sha(args.checkpoint),
        "policy_sha256": sha(model_path),
        "step_dt": STEP_DT,
        "observation_dim": 88,
        "history_steps": 5,
        "action_dim": 23,
        "proprioception_dim": 75,
        "command_dim": 11,
        "clock_dim": 2,
        "joint_names": JOINT_NAMES,
        "sdk_indices": SDK_INDICES,
        "sdk_joint_names": SDK_NAMES,
        "wrist_hold": WRIST_HOLD,
        "default_joint_pos": DEFAULT_Q,
        "kp": KP,
        "kd": KD,
        "action_scale": [ACTION_SCALE] * 23,
        "torque_limits": TORQUE_LIMITS,
        "joint_lower": JOINT_LOWER,
        "joint_upper": JOINT_UPPER,
        "obs_scales": OBS_SCALES,
        "command_scales": COMMAND_SCALES,
        "clip_observations": CLIP_OBSERVATIONS,
        "clip_actions": CLIP_ACTIONS,
        "arm_reference": {"model_indices": ARM_MODEL_INDICES, **ARM_REFERENCE},
        "imu_source": "pelvis",
        "torque_guard": True,
        "max_target_speed": 0.0,  # 0 = disabled; training has no target-speed limit
    }
    (out / "deploy.yaml").write_text(yaml.safe_dump(deploy_yaml, sort_keys=False), encoding="utf-8")
    manifest = {
        "format": FORMAT,
        "task": args.task,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": sha(args.checkpoint),
        "checkpoint_iteration": iteration,
        "source_commit": SOURCE_COMMITS[args.task],
        "policy_sha256": sha(model_path),
        "deploy_yaml_sha256": sha(out / "deploy.yaml"),
        "input_shape": [1, 5, 88],
        "output_shape": [1, 23],
        "joint_names": JOINT_NAMES,
        "sdk_indices": SDK_INDICES,
        "wrist_sdk_indices": WRIST_SDK_INDICES,
        "arm_model_indices": ARM_MODEL_INDICES,
        "default_joint_pos": DEFAULT_Q,
        "kp": KP, "kd": KD, "torque_limits": TORQUE_LIMITS,
        "command_scales": COMMAND_SCALES,
        "test_command": TEST_COMMAND,
        "onnx_opset": 17,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "python_version": platform.python_version(),
        "onnx_parity_samples": args.parity_samples,
        "onnx_max_abs_error": max_error,
        "onnx_realistic_max_abs_error": realistic_error,
        "onnx_parity_tolerance": "rtol=2e-4, atol=2e-4",
        "actor": {
            "model_type": "MlpAdaptModel",
            "is_recurrent": False,
            "mem_encoder": "375->256->128->32 (ELU)",
            "state_estimator": "32->64->32->3 (ELU)",
            "low_level_net": "123->256->128->32->23 (ELU)",
            "output": "deterministic mean action; checkpoint std not exported",
        },
        "notes": [
            "Observation scaling/clipping/history/arm reference are applied by the "
            "deployment core per the frozen training sources; the ONNX graph is pure actor.",
            "The exported graph contains no observation normalizer (HugWBC has none).",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

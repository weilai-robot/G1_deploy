#!/usr/bin/env python3
"""Replay the compiled holo_wbt controller inside MuJoCo.

The physics run here; observation assembly, ONNX inference, joint limits and
the rate-limited entry ramp all run in the C++ deployment binary, so this
exercises the same code that drives the robot instead of a Python imitation.

Each control period the driver sends q/dq/pelvis attitude/angular velocity and
the current reference frame to `holo_wbt_offline`, receives joint targets, and
turns them into torques with the deploy.yaml kp/kd (the MJCF actuators are
torque motors).

Usage:
    python tools/replay_holo_wbt_mujoco.py \
        --package deploy/robots/g1_29dof/config/policy/holo_wbt/czy_box \
        --offline build/holo_wbt_offline \
        --model  <path-to>/scene_g1_29dof_wbt_plane.xml \
        --start velocity --seconds 10 --report replay.json
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

FALL_HEIGHT = 0.35
FALL_TILT = 1.2


def load_package(package: Path) -> dict:
    config = yaml.safe_load((package / "deploy.yaml").read_text(encoding="utf-8"))
    if config["format"] != "holo_wbt_g1_154_v1":
        raise ValueError(f"unexpected package format {config['format']}")
    if len(config["joint_names"]) != config["action_dim"]:
        raise ValueError("joint_names/action_dim mismatch")
    return config


def load_reference_frame(package: Path, frame: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Joint positions plus pelvis attitude/height of one reference frame."""
    record_size = 2 * 29 + 7 * 14
    with (package / "reference.bin").open("rb") as handle:
        magic, frames, joints, _bodies, _dt = struct.unpack("<8sIIIf", handle.read(24))
        if magic != b"BMG1V1\0\0" or frame >= frames:
            raise ValueError("invalid reference table")
        handle.seek(24 + frame * record_size * 4)
        record = np.frombuffer(handle.read(record_size * 4), dtype="<f4").astype(np.float64)
    positions = record[2 * joints:2 * joints + 3 * 14].reshape(14, 3)
    quats = record[2 * joints + 3 * 14:].reshape(14, 4)
    return record[:joints].copy(), quats[0], float(positions[0, 2])


def yaw_aligned(quat: np.ndarray) -> np.ndarray:
    """Reference attitude with its absolute heading removed (robot starts facing +x)."""
    w, x, y, z = quat
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    rotated = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ rotation
    qw = np.sqrt(max(0.0, 1.0 + np.trace(rotated))) / 2.0
    qx = np.sqrt(max(0.0, 1.0 + rotated[0, 0] - rotated[1, 1] - rotated[2, 2])) / 2.0
    qy = np.sqrt(max(0.0, 1.0 - rotated[0, 0] + rotated[1, 1] - rotated[2, 2])) / 2.0
    qz = np.sqrt(max(0.0, 1.0 - rotated[0, 0] - rotated[1, 1] + rotated[2, 2])) / 2.0
    if rotated[2, 1] - rotated[1, 2] < 0:
        qx = -qx
    if rotated[0, 2] - rotated[2, 0] < 0:
        qy = -qy
    if rotated[1, 0] - rotated[0, 1] < 0:
        qz = -qz
    result = np.array([qw, qx, qy, qz])
    return result / np.linalg.norm(result)


class Replay:
    def __init__(self, args: argparse.Namespace) -> None:
        self.config = load_package(args.package)
        self.names = self.config["joint_names"]
        self.kp = np.asarray(self.config["kp"], dtype=np.float64)
        self.kd = np.asarray(self.config["kd"], dtype=np.float64)
        self.effort = np.asarray(self.config["effort_limit"], dtype=np.float64)
        self.default = np.asarray(self.config["default_joint_pos"], dtype=np.float64)
        self.dt = float(self.config["step_dt"])
        self.blend_s = args.blend_s
        self.blend_rate = args.blend_rate

        self.model = mujoco.MjModel.from_xml_path(str(args.model))
        self.model.opt.timestep = args.timestep
        self.decimation = max(1, int(round(self.dt / args.timestep)))
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 1280, 720

        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.names]
        actuators = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.names]
        if min(joints) < 0 or min(actuators) < 0:
            raise ValueError("MJCF is missing joints/actuators referenced by deploy.yaml")
        self.qpos_adr = np.array([self.model.jnt_qposadr[j] for j in joints])
        self.dof_adr = np.array([self.model.jnt_dofadr[j] for j in joints])
        self.actuator_ids = np.array(actuators)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.base_id < 0:
            raise ValueError("MJCF has no pelvis body")

        self.start_pose, self.start_quat, self.start_height = self._start_state(args)
        self.data = mujoco.MjData(self.model)
        self.offline = subprocess.Popen(
            [str(args.offline), str(args.package)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.trace: list[dict] = []
        self.frame = args.start_frame
        self.blend_tick = 0
        self.blending = self.blend_s > 0
        self.last_command = self.start_pose.copy()
        self.first = True
        self.fell = False
        self.hold = getattr(args, "hold", False)

    def _start_state(self, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
        """Initial joints and pelvis attitude/height for the chosen entry."""
        if args.start == "reference":
            joints, quat, height = load_reference_frame(args.package, args.start_frame)
            return joints, yaw_aligned(quat), height + args.height_offset
        return (np.asarray(self.config["default_joint_pos"], dtype=np.float64),
                np.array([1.0, 0.0, 0.0, 0.0]), args.height)

    def reset(self, height: float | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, self.start_height if height is None else height]
        self.data.qpos[3:7] = self.start_quat
        self.data.qpos[self.qpos_adr] = self.start_pose
        mujoco.mj_forward(self.model, self.data)

    def state_line(self) -> str:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 1)
        quat = self.data.xquat[self.base_id]
        payload = {
            "q": self.data.qpos[self.qpos_adr].tolist(),
            "dq": self.data.qvel[self.dof_adr].tolist(),
            "quat": [float(v) for v in quat],
            "omega": [float(v) for v in velocity[:3]],
            "frame": self.frame,
            "max_speed": self.config["max_target_speed"] if "max_target_speed" in self.config else 8.0,
        }
        if self.first:
            payload["reset"] = True
            self.first = False
        return json.dumps(payload)

    def control_step(self) -> dict:
        if self.hold:
            # 诊断用：不跑策略，直接保持初始姿态，检查物理与 PD 增益是否能支撑站立。
            reply = {"targets": self.start_pose.tolist(), "clipped": 0}
            command = self.start_pose
        else:
            self.offline.stdin.write(self.state_line() + "\n")
            self.offline.stdin.flush()
            line = self.offline.stdout.readline()
            if not line:
                error = self.offline.stderr.read().strip()
                raise RuntimeError(f"holo_wbt_offline stopped: {error}")
            reply = json.loads(line)
            policy_target = np.asarray(reply["targets"], dtype=np.float64)

            # 镜像 State_HoloWbt 的入场过渡：参考帧冻结，指令按速率上限逼近策略目标。
            command = policy_target
            if self.blending:
                step = self.blend_rate * self.dt
                delta = np.clip(policy_target - self.last_command, -step, step)
                command = self.last_command + delta
                remaining = float(np.abs(policy_target - command).max())
                self.blend_tick += 1
                if remaining < 0.05 or self.blend_tick > max(1, int(round(self.blend_s / self.dt))):
                    self.blending = False
            else:
                self.frame += 1
        self.last_command = command.copy()

        tilt = float(np.degrees(np.arccos(np.clip(self.data.xmat[self.base_id].reshape(3, 3)[2, 2], -1, 1))))
        self.trace.append({
            "t": round(len(self.trace) * self.dt, 3),
            "frame": self.frame,
            "blend": self.blending,
            "height": round(float(self.data.qpos[2]), 4),
            "tilt_deg": round(tilt, 2),
            "clipped": reply["clipped"],
            "target_error": round(float(np.abs(command - self.data.qpos[self.qpos_adr]).max()), 4),
        })

        for _ in range(self.decimation):
            error = command - self.data.qpos[self.qpos_adr]
            torque = np.clip(self.kp * error - self.kd * self.data.qvel[self.dof_adr], -self.effort, self.effort)
            self.data.ctrl[self.actuator_ids] = torque
            mujoco.mj_step(self.model, self.data)

        tilt_rad = np.arccos(np.clip(self.data.xmat[self.base_id].reshape(3, 3)[2, 2], -1, 1))
        if self.data.qpos[2] < FALL_HEIGHT or tilt_rad > FALL_TILT:
            self.fell = True
        return reply

    def close(self) -> None:
        if self.offline.poll() is None:
            self.offline.stdin.close()
            self.offline.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="scene_g1_29dof_wbt_plane.xml")
    parser.add_argument("--start", choices=("velocity", "reference"), default="velocity",
                        help="initial joint pose: Velocity stance or a reference frame")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="reference frame used for the initial pose and the first command")
    parser.add_argument("--height", type=float, default=0.76, help="initial pelvis height for --start velocity")
    parser.add_argument("--height-offset", type=float, default=0.0,
                        help="added to the reference pelvis height when --start reference")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--blend-s", type=float, default=2.5)
    parser.add_argument("--blend-rate", type=float, default=1.5)
    parser.add_argument("--hold", action="store_true",
                        help="diagnostic: hold the initial pose without running the policy")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    package = args.package.resolve()
    args.package = package
    replay = Replay(args)
    replay.reset(args.height)
    steps = int(round(args.seconds / replay.dt))
    started = time.perf_counter()
    try:
        for _ in range(steps):
            replay.control_step()
            if replay.fell:
                break
    finally:
        replay.close()

    trace = replay.trace
    fall = next((row for row in trace if row["height"] < FALL_HEIGHT or row["tilt_deg"] > np.degrees(FALL_TILT)), None)
    report = {
        "package": str(package),
        "start": args.start,
        "mujoco": mujoco.__version__,
        "physics_dt": args.timestep,
        "control_dt": replay.dt,
        "planned_seconds": args.seconds,
        "simulated_seconds": round(len(trace) * replay.dt, 3),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "fell": replay.fell,
        "fall_at": fall,
        "max_tilt_deg": max((row["tilt_deg"] for row in trace), default=0.0),
        "min_height": min((row["height"] for row in trace), default=0.0),
        "height_at_end": trace[-1]["height"] if trace else None,
        "blend_ticks": replay.blend_tick,
        "final_frame": trace[-1]["frame"] if trace else 0,
        "max_clipped": max((row["clipped"] for row in trace), default=0),
        "median_clipped_after_blend": float(np.median([r["clipped"] for r in trace if not r["blend"]] or [0])),
        "trace": trace[:: max(1, len(trace) // 40)],
    }
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not report["fell"]:
        print("REPLAY_OK: robot stayed upright for the whole segment")


if __name__ == "__main__":
    sys.exit(main())

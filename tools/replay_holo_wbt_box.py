#!/usr/bin/env python3
"""Closed-loop MuJoCo replay of the holo_wbt deployment binary with the training asset.

This mirrors holosoma's scripts/replay_g1_box_mujoco.py:
  * the MuJoCo model is compiled from the training collision URDF,
  * the robot and box start from the motion's first frame including velocities,
  * physics runs at the training rate and the policy acts every control_decimation,
and only the policy call differs: observations and joint targets come from the
compiled holo_wbt_offline binary instead of a PyTorch actor.

Usage:
    python tools/replay_holo_wbt_box.py \
        --package  .../config/policy/holo_wbt/czy_box \
        --offline  build/holo_wbt_offline \
        --urdf     .../data/robots/g1/main_mesh_collision_halfspherehand.urdf \
        --motion   .../data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz \
        --box      .../data/scene_objects/boxes/largebox.obj \
        --seconds 6.5 --report replay.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

MIN_STAND_HEIGHT = 0.35


def load_package(package: Path) -> dict:
    config = yaml.safe_load((package / "deploy.yaml").read_text(encoding="utf-8"))
    if config["format"] != "holo_wbt_g1_154_v1":
        raise ValueError(f"unexpected package format {config['format']}")
    return config


def load_motion(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


class Replay:
    def __init__(self, args: argparse.Namespace) -> None:
        self.config = load_package(args.package)
        self.names = [str(name) for name in self.config["joint_names"]]
        self.default = np.asarray(self.config["default_joint_pos"], dtype=np.float64)
        self.kp = np.asarray(self.config["kp"], dtype=np.float64)
        self.kd = np.asarray(self.config["kd"], dtype=np.float64)
        self.limits = np.asarray(self.config["effort_limit"], dtype=np.float64)
        self.scale = np.asarray(self.config["action_scale"], dtype=np.float64)
        self.dt = float(self.config["step_dt"])

        motion = load_motion(args.motion)
        self.motion = motion
        motion_names = [str(name) for name in motion["joint_names"].tolist()]
        if sorted(motion_names) != sorted(self.names):
            raise ValueError("motion joint names differ from the deploy package")
        self.motion_ids = [motion_names.index(name) for name in self.names]
        body_names = [str(name) for name in motion["body_names"].tolist()]
        self.ref_body_name = "torso_link"
        self.motion_ref_id = body_names.index(self.ref_body_name)
        self.motion_root_id = body_names.index("pelvis")
        self.frames = int(motion["joint_pos"].shape[0])
        self.start = args.start_frame

        spec = mujoco.MjSpec.from_file(str(args.urdf))
        spec.body("pelvis").add_freejoint(name="robot_free")
        spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1],
            rgba=[0.22, 0.28, 0.35, 1], friction=[1, 0.005, 0.0001],
        )
        spec.worldbody.add_light(pos=[1, -2, 4], dir=[-0.2, 0.2, -1])
        if args.box is not None:
            spec.add_mesh(name="carry_box_mesh", file=str(args.box.resolve()))
            box = spec.worldbody.add_body(name="carry_box")
            box.add_freejoint(name="box_free")
            box.add_geom(name="carry_box_collision", type=mujoco.mjtGeom.mjGEOM_MESH,
                         meshname="carry_box_mesh", mass=args.mass, rgba=[0.85, 0.55, 0.25, 1],
                         friction=[0.35, 0.005, 0.0001])
        for name, limit in zip(self.names, self.limits):
            spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              ctrllimited=True, ctrlrange=[-limit, limit])
        self.model = spec.compile()
        self.model.opt.timestep = 1.0 / args.physics_hz
        self.decimation = max(1, int(round(self.dt * args.physics_hz)))

        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.names]
        if min(joint_ids) < 0:
            raise ValueError("joints missing from the training URDF")
        self.qpos_ids = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        self.qvel_ids = np.array([self.model.jnt_dofadr[j] for j in joint_ids])
        self.actuator_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                                      for name in self.names])
        self.root_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.ref_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ref_body_name)
        self.box_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "carry_box")
        self.data = mujoco.MjData(self.model)
        self.offline = subprocess.Popen(
            [str(args.offline), str(args.package)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.frame = self.start
        self.action = np.zeros(len(self.names))
        self.trace: list[dict] = []
        self.first = True
        self.fell = False
        self.debug = args.debug
        self.zero_base_velocity = args.zero_base_velocity
        self._initialise_state()

    def _initialise_state(self) -> None:
        motion = self.motion
        self.data.qpos[:7] = motion["joint_pos"][self.start, :7]
        self.data.qpos[self.qpos_ids] = motion["joint_pos"][self.start, 7:][self.motion_ids]
        # NPZ stores MuJoCo free-joint velocities: world linear, local angular.
        self.data.qvel[:6] = 0.0 if self.zero_base_velocity else motion["joint_vel"][self.start, :6]
        self.data.qvel[self.qvel_ids] = motion["joint_vel"][self.start, 6:][self.motion_ids]
        if self.box_body >= 0:
            box_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_free")
            pos_id, vel_id = self.model.jnt_qposadr[box_joint], self.model.jnt_dofadr[box_joint]
            self.data.qpos[pos_id:pos_id + 3] = motion["object_pos_w"][self.start]
            self.data.qpos[pos_id + 3:pos_id + 7] = motion["object_quat_w"][self.start]
            self.data.qvel[vel_id:vel_id + 3] = motion["object_lin_vel_w"][self.start]
        mujoco.mj_forward(self.model, self.data)

    def control_step(self) -> None:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.root_body, velocity, 1)
        payload = {
            "q": self.data.qpos[self.qpos_ids].tolist(),
            "dq": self.data.qvel[self.qvel_ids].tolist(),
            "quat": [float(v) for v in self.data.xquat[self.root_body]],
            "omega": [float(v) for v in velocity[:3]],
            "frame": self.frame,
            "max_speed": 8.0,
        }
        if self.first:
            payload["reset"] = True
            self.first = False
        self.offline.stdin.write(json.dumps(payload) + "\n")
        self.offline.stdin.flush()
        line = self.offline.stdout.readline()
        if not line:
            raise RuntimeError("holo_wbt_offline stopped: " + self.offline.stderr.read().strip())
        reply = json.loads(line)
        target = np.asarray(reply["targets"], dtype=np.float64)
        if self.debug and len(self.trace) < 3:
            obs = np.asarray(reply["obs"], dtype=np.float64)
            print("DEBUG step", len(self.trace))
            print("  omega", np.round(payload["omega"], 3), "quat", np.round(payload["quat"], 3))
            print("  obs   actions[:3]", np.round(obs[:3], 3), "base_ang_vel", np.round(obs[29:32], 3),
                  "dof_pos[:3]", np.round(obs[32:35], 3), "ori", np.round(obs[148:154], 3))
            print("  action max", round(float(np.abs(reply["actions"]).max()), 2),
                  "clipped", reply["clipped"],
                  "| target-q max", round(float(np.abs(target - self.data.qpos[self.qpos_ids]).max()), 3))
            print("  sim ncon", self.data.ncon,
                  "max|qvel|", round(float(np.abs(self.data.qvel).max()), 2),
                  "penetration", round(float(min([c.dist for c in self.data.contact[: self.data.ncon]] or [0.0])), 4),
                  "ctrl max", round(float(np.abs(self.data.ctrl).max()), 1))
        for _ in range(self.decimation):
            torque = self.kp * (target - self.data.qpos[self.qpos_ids]) - self.kd * self.data.qvel[self.qvel_ids]
            self.data.ctrl[self.actuator_ids] = np.clip(torque, -self.limits, self.limits)
            mujoco.mj_step(self.model, self.data)
        self.action = np.asarray(reply["actions"], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        tilt = float(np.degrees(np.arccos(np.clip(self.data.xmat[self.root_body].reshape(3, 3)[2, 2], -1, 1))))
        self.trace.append({
            "t": round(len(self.trace) * self.dt, 3),
            "frame": self.frame,
            "height": round(float(self.data.qpos[2]), 4),
            "tilt_deg": round(tilt, 2),
            "clipped": reply["clipped"],
        })
        self.frame += 1
        # 只看是否真的倒下：起始姿态本身前倾 26° 且带 2.5 rad/s 角速度，单看倾角会误判。
        if self.data.qpos[2] < MIN_STAND_HEIGHT:
            self.fell = True

    def close(self) -> None:
        if self.offline.poll() is None:
            self.offline.stdin.close()
            self.offline.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--box", type=Path, default=None, help="largebox.obj; omit for a box-free rollout")
    parser.add_argument("--mass", type=float, default=2.6)
    parser.add_argument("--physics-hz", type=float, default=200.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=6.5)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", help="print the first policy exchanges")
    parser.add_argument("--zero-base-velocity", action="store_true",
                        help="start with a still pelvis instead of the motion's frame velocity")
    parser.add_argument("--video", type=Path, default=None, help="write an mp4 of the rollout")
    parser.add_argument("--video-fps", type=int, default=25)
    args = parser.parse_args()

    replay = Replay(args)
    steps = int(round(args.seconds / replay.dt))
    started = time.perf_counter()
    writer = renderer = camera = None
    if args.video is not None:
        import cv2

        args.video.parent.mkdir(parents=True, exist_ok=True)
        # 离屏渲染的帧缓冲默认 640x480，先放大再创建渲染器。
        replay.model.vis.global_.offwidth = 960
        replay.model.vis.global_.offheight = 540
        renderer = mujoco.Renderer(replay.model, height=540, width=960)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (960, 540))
        if not writer.isOpened():
            raise RuntimeError("video writer did not open")
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.0, 135, -18
    record_every = max(1, int(round(1.0 / args.video_fps / replay.dt)))
    try:
        for step in range(steps):
            if writer is not None and step % record_every == 0:
                camera.lookat[:] = replay.data.xpos[replay.root_body]
                camera.lookat[2] = 0.6
                renderer.update_scene(replay.data, camera=camera)
                frame = renderer.render()
                import cv2

                cv2.putText(frame, f"t={step * replay.dt:.2f}s  frame={replay.frame}", (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if step * replay.dt >= 3.0 and replay.fell:
                break  # 已倒地，画面没有更多信息
            replay.control_step()
    finally:
        replay.close()
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()

    trace = replay.trace
    report = {
        "package": str(args.package.resolve()),
        "mujoco": mujoco.__version__,
        "physics_hz": args.physics_hz,
        "control_dt": replay.dt,
        "start_frame": args.start_frame,
        "planned_seconds": args.seconds,
        "simulated_seconds": round(len(trace) * replay.dt, 3),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "fell": replay.fell,
        "max_tilt_deg": max((row["tilt_deg"] for row in trace), default=0.0),
        "min_height": min((row["height"] for row in trace), default=0.0),
        "height_at_end": trace[-1]["height"] if trace else None,
        "max_clipped": max((row["clipped"] for row in trace), default=0),
        "frames_played": len(trace),
        "trace": trace[:: max(1, len(trace) // 40)],
    }
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not replay.fell:
        print("REPLAY_OK: the deployment binary kept the robot upright for the whole segment")


if __name__ == "__main__":
    main()

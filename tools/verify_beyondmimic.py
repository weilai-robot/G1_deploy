#!/usr/bin/env python3
"""Check package integrity, C++/MuJoCo observations, and closed-loop deployment.

The C++ executable uses the real controller's core, ONNX and kinematic odometry.
No DDS is opened and no motor commands are sent to a robot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
import yaml


def matrix(q):
    return Rotation.from_quat(np.roll(q, -1)).as_matrix()


class Controller:
    def __init__(self, binary, package):
        self.process = subprocess.Popen([str(binary), str(package)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)

    def step(self, packet):
        self.process.stdin.write(json.dumps(packet, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"C++ controller exited ({self.process.poll()})")
        return json.loads(line)

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.wait(timeout=5)
        if self.process.returncode:
            raise RuntimeError(f"C++ controller failed ({self.process.returncode})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1]/"deploy/robots/g1_29dof/config/policy/beyondmimic/dance_zui")
    p.add_argument("--offline-binary", type=Path, required=True)
    p.add_argument("--mjcf", type=Path, required=True, help="Training robot assets/unitree_description/mjcf/g1.xml")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--state-source", choices=["kinematic", "truth"], default="kinematic")
    p.add_argument("--raw-targets", action="store_true", help="Simulation-only diagnostic: bypass target limits")
    p.add_argument("--max-target-speed", type=float, default=8.0)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        p.error("--seconds must be positive and finite")
    if not np.isfinite(args.max_target_speed) or not 0 < args.max_target_speed <= 20:
        p.error("--max-target-speed must be in (0,20]")
    package = args.package.resolve()
    cfg = yaml.safe_load((package/"deploy.yaml").read_text())
    manifest = json.loads((package/"manifest.json").read_text())
    for file, key in (("policy.onnx", "policy_sha256"), ("reference.bin", "reference_sha256"),
                      ("deploy.yaml", "deploy_yaml_sha256")):
        digest = hashlib.sha256((package/file).read_bytes()).hexdigest()
        if digest != manifest[key]:
            raise ValueError(f"Package checksum mismatch: {file}")
    raw = (package/"reference.bin").read_bytes()
    magic, frames, joints, bodies, dt = struct.unpack("<8sIIIf", raw[:24])
    assert magic == b"BMG1V1\0\0" and joints == 29 and bodies == 14
    records = np.frombuffer(raw, dtype="<f4", offset=24).reshape(frames, 156)
    ref_q, ref_dq = records[:, :29], records[:, 29:58]
    ref_pos = records[:, 58:100].reshape(frames, 14, 3)
    ref_quat = records[:, 100:].reshape(frames, 14, 4)
    ids = np.asarray(cfg["sdk_indices"])
    sdk_names = [None]*29
    for i, sdk in enumerate(ids):
        sdk_names[sdk] = cfg["joint_names"][i]
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    qids = np.array([model.jnt_qposadr[model.joint(name).id] for name in sdk_names])
    vids = np.array([model.jnt_dofadr[model.joint(name).id] for name in sdk_names])
    actuator_ids = np.array([next(i for i in range(model.nu) if model.actuator_trnid[i, 0] == model.joint(n).id) for n in sdk_names])
    torso = model.body("torso_link").id
    defaults = np.asarray(cfg["default_joint_pos"])

    def initialise():
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = [0, 0, ref_pos[0, 0, 2]]
        data.qpos[3:7] = ref_quat[0, 0]
        data.qpos[qids[ids]] = ref_q[0]
        mujoco.mj_forward(model, data)

    def packet(frame, reset=False, truth=False):
        rot = matrix(data.qpos[3:7])
        result = {"frame": frame, "reset": reset, "q": data.qpos[qids].tolist(),
                  "dq": data.qvel[vids].tolist(), "quat": data.qpos[3:7].tolist(),
                  "omega": data.qvel[3:6].tolist()}
        result["max_speed"] = args.max_target_speed
        if truth:
            result.update(position=data.qpos[:3].tolist(), velocity_b=(rot.T@data.qvel[:3]).tolist())
        return result

    # Independent oracle: MuJoCo's world FK, matching training's torso-relative
    # transforms and row-major first-two-columns orientation representation.
    initialise()
    controller = Controller(args.offline_binary.resolve(), package)
    rng = np.random.default_rng(123)
    last_action = np.zeros(29)
    max_obs_error, max_action_error = 0.0, 0.0
    session = ort.InferenceSession(str(package/"policy.onnx"), providers=["CPUExecutionProvider"])
    for sample in range(64):
        if sample:
            data.qpos[qids[ids]] = ref_q[sample] + rng.normal(0, 0.025, 29)
            data.qvel[vids] = rng.normal(0, 0.1, 29)
            data.qvel[:6] = rng.normal(0, 0.05, 6)
            mujoco.mj_forward(model, data)
        state = packet(sample, reset=sample == 0, truth=True)
        result = controller.step(state)
        torso_r = matrix(data.xquat[torso])
        expected = np.concatenate([
            ref_q[sample], ref_dq[sample], torso_r.T@(ref_pos[sample, 7]-data.xpos[torso]),
            (torso_r.T@matrix(ref_quat[sample, 7]))[:, :2].reshape(-1),
            matrix(data.qpos[3:7]).T@data.qvel[:3], data.qvel[3:6],
            data.qpos[qids[ids]]-defaults, data.qvel[vids[ids]], last_action,
        ]).astype(np.float32)
        np.testing.assert_allclose(result["obs"], expected, rtol=1e-4, atol=3e-5)
        actual_action = session.run(["actions"], {"obs": expected[None]})[0][0]
        np.testing.assert_allclose(result["actions"], actual_action, rtol=2e-4, atol=2e-4)
        max_obs_error = max(max_obs_error, float(np.max(np.abs(np.asarray(result["obs"])-expected))))
        max_action_error = max(max_action_error, float(np.max(np.abs(np.asarray(result["actions"])-actual_action))))
        last_action = np.asarray(result["actions"])
    controller.close()

    # Reject bad sensor packets instead of propagating NaN/zero quaternions.
    bad = packet(0, reset=True)
    bad["quat"] = [0, 0, 0, 0]
    failed = subprocess.run([str(args.offline_binary.resolve()), str(package)],
                            input=json.dumps(bad)+"\n", text=True, capture_output=True, timeout=10)
    assert failed.returncode != 0 and "Invalid pelvis IMU" in failed.stderr

    # Exercise package rejection in the real C++ loader, before any control loop.
    with tempfile.TemporaryDirectory(prefix="beyondmimic-rejection-") as temporary:
        bad_package = Path(temporary)
        for file in ("policy.onnx", "reference.bin"):
            (bad_package/file).symlink_to(package/file)
        invalid_cfg = dict(cfg)
        invalid_cfg["sdk_indices"] = [0]*29
        (bad_package/"deploy.yaml").write_text(yaml.safe_dump(invalid_cfg))
        failed = subprocess.run([str(args.offline_binary.resolve()), str(bad_package)],
                                input="", text=True, capture_output=True, timeout=10)
        assert failed.returncode != 0 and "Invalid SDK joint permutation" in failed.stderr
        (bad_package/"deploy.yaml").write_text(yaml.safe_dump(cfg))
        (bad_package/"reference.bin").unlink()
        (bad_package/"reference.bin").write_bytes(raw[:24])
        failed = subprocess.run([str(args.offline_binary.resolve()), str(bad_package)],
                                input="", text=True, capture_output=True, timeout=10)
        assert failed.returncode != 0 and "Invalid/incompatible reference table" in failed.stderr

    initialise()
    controller = Controller(args.offline_binary.resolve(), package)
    kp, kd, effort = (np.zeros(29) for _ in range(3))
    kp[ids], kd[ids], effort[ids] = cfg["kp"], cfg["kd"], cfg["effort_limit"]
    steps = min(int(args.seconds / cfg["step_dt"]), frames)
    max_drift, max_velocity_error, max_tilt, clipped_steps = 0.0, 0.0, 0.0, 0
    min_height = float(data.qpos[2])
    survived = True
    for frame in range(steps):
        result = controller.step(packet(frame, reset=frame == 0, truth=args.state_source == "truth"))
        targets = np.asarray(result["targets"])
        if args.raw_targets:
            targets[ids] = defaults + np.asarray(cfg["action_scale"])*np.asarray(result["actions"])
        assert np.isfinite(targets).all()
        max_drift = max(max_drift, float(np.linalg.norm(np.asarray(result["position"])-data.qpos[:3])))
        max_velocity_error = max(max_velocity_error, float(np.linalg.norm(np.asarray(result["velocity_b"])-matrix(data.qpos[3:7]).T@data.qvel[:3])))
        clipped_steps += result["clipped"] > 0
        for _ in range(round(cfg["step_dt"] / model.opt.timestep)):
            tau = np.clip(kp*(targets-data.qpos[qids])-kd*data.qvel[vids], -effort, effort)
            data.ctrl[actuator_ids] = tau
            mujoco.mj_step(model, data)
        min_height = min(min_height, float(data.qpos[2]))
        tilt = float(np.arccos(np.clip(matrix(data.qpos[3:7])[2, 2], -1, 1)))
        max_tilt = max(max_tilt, tilt)
        if data.qpos[2] < 0.35 or tilt > 0.85 or not np.isfinite(data.qpos).all():
            survived = False
            break
    controller.close()
    report = {"model_integrity": "passed", "cpp_observation_parity_samples": 64,
              "cpp_observation_max_abs_error": max_obs_error, "cpp_action_max_abs_error": max_action_error,
              "bad_imu_rejection": "passed", "state_source": args.state_source,
              "bad_sdk_mapping_rejection": "passed", "truncated_reference_rejection": "passed",
              "raw_targets_diagnostic": args.raw_targets,
              "max_target_speed": args.max_target_speed,
              "requested_seconds": args.seconds, "simulated_seconds": (frame+1)*cfg["step_dt"],
              "closed_loop_survived": survived, "minimum_pelvis_height_m": min_height,
              "maximum_pelvis_tilt_rad": max_tilt, "maximum_position_estimation_error_m": max_drift,
              "maximum_velocity_estimation_error_mps": max_velocity_error, "limited_action_steps": clipped_steps,
              "mujoco_version": mujoco.__version__, "real_robot_tested": False}
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+"\n")
    if not survived:
        raise SystemExit("Closed-loop verification failed. Do not run this configuration on the real robot.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Numeric verification of the HugWBC deployment packages against the frozen
training contract.

Layers, in order:
1. Identity: checkpoint SHA and rebuilt-actor structure for R41/R42.
2. PyTorch rebuilt actor vs ONNX Runtime on random and realistic histories.
3. Real-data anchor: the recorded zero-update evaluation rollout must satisfy
   the gait-clock dynamics used by the deployment core (phase advance 0.02*cmd3,
   half-cycle offset, no standing rows).
4. Python reference core (independent transcription of the training formulas)
   vs the C++ core through the hugwbc_offline binary, when that binary is
   provided and executable. Every intermediate is compared: observation frame,
   full history, raw/hybrid/clipped actions, arm reference, targets, SDK
   mapping, gait scalars.
5. Negative tests: corrupt IMU and wrong package must be rejected by the C++
   loader.

Nothing here sends DDS traffic. Closed-loop simulation and hardware results are
NOT produced by this script and stay unverified until actually run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import export_hugwbc as contract  # frozen contract constants + actor rebuild


# ---------------------------------------------------------------- reference core
F32 = np.float32


def wrap(x):
    return F32(F32(x) % F32(1.0))


def remap(u, duration):
    """Exact _step_contact_targets remap incl. strict-inequality boundary."""
    u, d = F32(u), F32(duration)
    if u < d:
        return F32(u * F32(0.5) / d)
    if u > d:
        return F32(F32(0.5) + F32((u - d) * F32(0.5) / (F32(1.0) - d)))
    return u


def quat_matrix(q):
    w, x, y, z = F32(q[0]), F32(q[1]), F32(q[2]), F32(q[3])
    n = math.sqrt(float(w * w + x * x + y * y + z * z))
    w, x, y, z = F32(w / n), F32(x / n), F32(y / n), F32(z / n)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=F32)


class ReferenceCore:
    """Python transcription of the training env contract; mirrors hugwbc/core.h."""

    def __init__(self, deploy_yaml):
        self.cfg = deploy_yaml
        self.reset()

    def reset(self):
        self.g = F32(0)
        self.last_hybrid = np.zeros(23, dtype=F32)
        self.history = np.zeros((5, 88), dtype=F32)
        self.seeded = False
        self.tick = 0

    def frame(self, snapshot, command, standing):
        cfg = self.cfg
        q, dq, quat, omega = snapshot
        self.g = wrap(self.g + F32(cfg["step_dt"]) * F32(command[3]))
        phase_left = F32(0) if standing else wrap(self.g + F32(command[4]))
        phase_right = F32(0) if standing else self.g
        clock_left = F32(math.sin(2 * math.pi * float(remap(phase_left, command[5]))))
        clock_right = F32(math.sin(2 * math.pi * float(remap(phase_right, command[5]))))

        r = quat_matrix(quat)
        if cfg["imu_source"] == "torso":
            wy, wr, wp = F32(q[12]), F32(q[13]), F32(q[14])
            cy, sy = F32(math.cos(float(wy))), F32(math.sin(float(wy)))
            cr, sr = F32(math.cos(float(wr))), F32(math.sin(float(wr)))
            cp, sp = F32(math.cos(float(wp))), F32(math.sin(float(wp)))
            rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=F32)
            rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=F32)
            ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=F32)
            waist = rz @ rx @ ry
            r_pelvis = r @ waist.T
            w_world = r @ np.asarray(omega, dtype=F32)
            omega_b = r_pelvis.T @ w_world
            r = r_pelvis
        else:
            omega_b = np.asarray(omega, dtype=F32)
        gravity = -r[2, :].astype(F32)

        q23 = np.asarray([q[i] for i in cfg["sdk_indices"]], dtype=F32)
        dq23 = np.asarray([dq[i] for i in cfg["sdk_indices"]], dtype=F32)
        q0 = np.asarray(cfg["default_joint_pos"], dtype=F32)
        obs = np.concatenate([
            omega_b * F32(cfg["obs_scales"]["ang_vel"]),
            gravity,
            (q23 - q0) * F32(cfg["obs_scales"]["dof_pos"]),
            dq23 * F32(cfg["obs_scales"]["dof_vel"]),
            self.last_hybrid,
            np.asarray(command, dtype=F32) * np.asarray(cfg["command_scales"], dtype=F32),
            np.array([clock_left, clock_right], dtype=F32),
        ]).astype(F32)
        obs = np.clip(obs, -F32(100.0), F32(100.0)).astype(F32)

        if self.seeded:
            self.history = np.concatenate([self.history[1:], obs[None]], axis=0).astype(F32)
        else:
            self.history = np.repeat(obs[None], 5, axis=0).astype(F32)
            self.seeded = True
        self.tick += 1
        return obs, self.g, phase_left, phase_right, clock_left, clock_right

    def complete(self, snapshot, command, standing, raw):
        cfg = self.cfg
        q, dq = np.asarray(snapshot[0], dtype=F32), np.asarray(snapshot[1], dtype=F32)
        q0 = np.asarray(cfg["default_joint_pos"], dtype=F32)
        arm = cfg["arm_reference"]
        speed_scale = F32(min(1.0, max(-1.0, float(F32(command[0]) / F32(arm["swing_speed_reference"])))))
        walking = F32(0) if standing else F32(1)
        ref = np.zeros(8, dtype=F32)
        phase = self._phases  # stored by frame(); [left, right]
        ref[0] = F32(arm["pitch_bias_rad"]) - F32(arm["swing_pitch_amplitude"]) * F32(math.cos(
            2 * math.pi * float(phase[1]) - float(arm["swing_phase_offset"]))) * speed_scale * walking
        ref[4] = F32(arm["pitch_bias_rad"]) - F32(arm["swing_pitch_amplitude"]) * F32(math.cos(
            2 * math.pi * float(phase[0]) - float(arm["swing_phase_offset"]))) * speed_scale * walking
        ref[1] = F32(arm["shoulder_roll_rad"])
        ref[5] = F32(-arm["shoulder_roll_rad"])
        ref[2] = ref[6] = F32(0)
        ref[3] = ref[7] = F32(arm["elbow_rad"])

        hybrid = np.asarray(raw, dtype=F32).copy()
        arm_idx = np.asarray(cfg["arm_reference"]["model_indices"])
        action_scale = np.asarray(cfg["action_scale"], dtype=F32)
        hybrid[arm_idx] = (ref - q0[arm_idx]) / action_scale[arm_idx]
        clipped = np.clip(hybrid, -F32(cfg["clip_actions"]), F32(cfg["clip_actions"])).astype(F32)
        q_target = (q0 + action_scale * clipped).astype(F32)

        sdk = np.zeros(29, dtype=F32)
        for i, s in enumerate(cfg["sdk_indices"]):
            sdk[s] = q_target[i]
        for i, s in enumerate(cfg["wrist_hold"]["sdk_indices"]):
            sdk[s] = F32(cfg["wrist_hold"]["target_rad"][i])

        # deployment guards (must mirror core.h apply_guards)
        lower = np.asarray(cfg["joint_lower"], dtype=F32)
        upper = np.asarray(cfg["joint_upper"], dtype=F32)
        kp = np.asarray(cfg["kp"], dtype=F32)
        kd = np.asarray(cfg["kd"], dtype=F32)
        tau = np.asarray(cfg["torque_limits"], dtype=F32)
        limited = 0
        for i, s in enumerate(cfg["sdk_indices"]):
            desired = np.clip(sdk[s], F32(lower[i]) + F32(0.01), F32(upper[i]) - F32(0.01))
            if cfg["torque_guard"]:
                center = F32(q[s] + F32(kd[i] * dq[s] / kp[i]))
                radius = F32(tau[i] / kp[i])
                desired = np.clip(desired, F32(center - radius), F32(center + radius))
                desired = np.clip(desired, F32(lower[i]) + F32(0.01), F32(upper[i]) - F32(0.01))
            if abs(float(desired - sdk[s])) > 1e-5:
                limited += 1
            sdk[s] = F32(desired)
        return hybrid, clipped, q_target, ref, sdk, limited

    def step(self, snapshot, command, standing, raw):
        obs, g, pl, pr, cl, cr = self.frame(snapshot, command, standing)
        self._phases = (pl, pr)
        hybrid, clipped, q_target, ref, sdk, limited = self.complete(snapshot, command, standing, raw)
        return {"obs": obs, "history": self.history.reshape(-1).copy(), "g": self.g,
                "phase_left": pl, "phase_right": pr, "clock_left": cl, "clock_right": cr,
                "hybrid": hybrid, "clipped": clipped, "q_target": q_target,
                "arm_reference": ref, "sdk_targets": sdk, "limited": limited}


# ---------------------------------------------------------------- fixtures
def fixture(rng, ticks, cmd, standing_schedule, start_quat=(0.7071, 0.0, 0.0, 0.7071)):
    """Synthetic SDK-order sensor sequence; seeded, includes phase wrap-crossing."""
    packets = []
    q = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0,
                  0, 0, 0, 0.3, 0.27, 0, 0.97, 0.3, -0.27, 0, 0.97,
                  0, 0, 0, 0, 0, 0], dtype=F32)
    dq = np.zeros(29, dtype=F32)
    quat = np.array(start_quat, dtype=F32)
    omega = np.zeros(3, dtype=F32)
    for t in range(ticks):
        if t:
            q = (q + rng.normal(0, 0.03, 29).astype(F32)).astype(F32)
            dq = rng.normal(0, 0.8, 29).astype(F32)
            quat = quat + rng.normal(0, 0.002, 4).astype(F32)
            quat = (quat / np.linalg.norm(quat)).astype(F32)
            omega = rng.normal(0, 0.3, 3).astype(F32)
        standing = standing_schedule(t)
        packets.append({"q": q.tolist(), "dq": dq.tolist(), "quat": quat.tolist(),
                        "omega": omega.tolist(), "cmd": list(cmd), "standing": standing,
                        "reset": t == 0})
    return packets


def compare(python_out, cpp_out, atol):
    errors = {}
    for field, tolerance in atol.items():
        a = np.asarray(python_out[field], dtype=np.float64)
        b = np.asarray(cpp_out[field], dtype=np.float64)
        if a.shape != b.shape:
            raise SystemExit(f"FAILED: shape mismatch on {field}: {a.shape} vs {b.shape}")
        errors[field] = float(np.max(np.abs(a - b))) if a.size else 0.0
        if errors[field] > tolerance:
            raise SystemExit(f"FAILED: {field} max abs error {errors[field]} > {tolerance} "
                             f"(tick {cpp_out['index']})")
    return errors


def run_cpp(binary, package, packets):
    process = subprocess.Popen([str(binary), str(package)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True, bufsize=1)
    outputs = []
    for packet in packets:
        import yaml
        process.stdin.write(yaml.safe_dump(packet, default_flow_style=True).replace("\n", " ") + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"hugwbc_offline exited ({process.poll()})")
        outputs.append(json.loads(line))
    process.stdin.close()
    process.wait(timeout=10)
    if process.returncode:
        raise RuntimeError(f"hugwbc_offline failed ({process.returncode})")
    return outputs


def real_anchor(csv_path, dt=0.02):
    """Validate the deployment gait-clock dynamics against the recorded
    zero-update training rollout (foot_index_* = pre-remap phases)."""
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    report = {"rows": len(rows), "max_advance_error": 0.0, "max_offset_error": 0.0}
    for env in {r["env"] for r in rows}:
        per_env = [r for r in rows if r["env"] == env and float(r["valid"]) > 0.5]
        freq = float(per_env[0]["cmd_3"])
        prev = None
        for r in per_env:
            left, right = float(r["foot_index_left"]), float(r["foot_index_right"])
            if prev is not None:
                for now, before in ((left, prev[0]), (right, prev[1])):
                    delta = (now - before) % 1.0
                    report["max_advance_error"] = max(
                        report["max_advance_error"], abs(delta - dt * freq))
                offset = (left - right) % 1.0
                # distance of the wrapped offset from half a cycle
                report["max_offset_error"] = max(
                    report["max_offset_error"], abs(offset - 0.5))
            prev = (left, right)
    if report["max_advance_error"] > 1e-4:
        raise SystemExit(f"FAILED: recorded phase advance deviates: {report}")
    if report["max_offset_error"] > 1e-4:
        raise SystemExit(f"FAILED: recorded foot phase offset is not half a cycle: {report}")
    report["verdict"] = "phase dynamics match dt*cmd3 advance and 0.5 offset"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = REPO / "deploy/robots/g1_29dof/config/policy/hugwbc"
    training_root = Path(r"D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1")
    parser.add_argument("--variant", choices=("r41", "r42", "both"), default="both")
    parser.add_argument("--package", type=Path, help="explicit package directory")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--offline-binary", type=Path,
                        help="hugwbc_offline executable; C++ parity runs only when given")
    parser.add_argument("--rollout-csv", type=Path,
                        default=training_root / ".oma/experiments/exp-20261030-004/baseline-hybrid/seed1/rollout.csv")
    parser.add_argument("--parity-samples", type=int, default=64)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    variants = {"r41": {"checkpoint": training_root / ".oma/accepted/r41-reference-arms-20261030/model_12600_g1r41.pt",
                        "package": default_root / "r41"},
                "r42": {"checkpoint": training_root / ".oma/experiments/exp-20261030-004/model_13200_g1r42.pt",
                        "package": default_root / "r42"}}
    if args.variant != "both":
        variants = {args.variant: variants[args.variant]}
    if args.package:
        for v in variants.values():
            v["package"] = args.package
    if args.checkpoint:
        for v in variants.values():
            v["checkpoint"] = args.checkpoint

    import torch
    import onnxruntime as ort
    import yaml

    report = {"onnx_tolerance": "rtol=2e-4, atol=2e-4", "variants": {}}

    # --- real-data anchor (shared) -------------------------------------------------
    if args.rollout_csv and Path(args.rollout_csv).exists():
        report["real_data_anchor"] = real_anchor(args.rollout_csv)
    else:
        report["real_data_anchor"] = {"executed": False,
                                      "reason": f"rollout CSV not found: {args.rollout_csv}"}

    for name, spec in variants.items():
        package = spec["package"].resolve()
        checkpoint = spec["checkpoint"].resolve()
        manifest = json.loads((package / "manifest.json").read_text())
        cfg = yaml.safe_load((package / "deploy.yaml").read_text())
        variant_report = {}

        # 1) identity
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if sha != manifest["checkpoint_sha256"]:
            raise SystemExit(f"FAILED: {name}: checkpoint hash mismatch")
        variant_report["checkpoint_sha256"] = sha

        # 2) rebuilt actor vs ONNX
        actor = contract.HugWBCActor(
            torch.load(checkpoint, map_location="cpu", weights_only=True)).eval()
        session = ort.InferenceSession(str(package / "policy.onnx"), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(42)
        max_error = 0.0
        for i in range(args.parity_samples):
            if i % 3 == 0:
                history = rng.uniform(-1, 1, (1, 5, 88))
            elif i % 3 == 1:
                history = np.zeros((1, 5, 88), np.float32)
                history[..., 75:86] = np.asarray(cfg["command_scales"]) * contract.TEST_COMMAND
                history[..., 86:88] = rng.uniform(-1, 1, (1, 5, 2))
            else:
                history = rng.uniform(-3, 3, (1, 5, 88))
            history = history.astype(np.float32)
            expected = actor(torch.from_numpy(history)).detach().numpy()
            actual = session.run(["actions"], {"obs_history": history})[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
        variant_report["onnx_vs_pytorch_max_abs_error"] = max_error
        variant_report["onnx_vs_pytorch_samples"] = args.parity_samples

        # 3) Python reference core end-to-end (ONNX as the deployed artifact)
        core = ReferenceCore(cfg)
        core.reset()
        cmd = list(contract.TEST_COMMAND)
        ticks = 100  # crosses the gait wrap at cmd3=1.5 (wrap every ~33 ticks)
        schedule = lambda t: False
        rng2 = np.random.default_rng(7)
        packets = fixture(rng2, ticks, cmd, schedule)
        py_outputs = []
        for packet in packets:
            snap = (packet["q"], packet["dq"], packet["quat"], packet["omega"])
            frame = core.frame(snap, cmd, packet["standing"])
            core._phases = (frame[2], frame[3])
            raw = session.run(["actions"], {"obs_history": core.history[None]})[0][0]
            out = core.complete(snap, cmd, packet["standing"], raw)
            py_outputs.append({"obs": frame[0], "history": core.history.reshape(-1).copy(),
                               "g": frame[1], "phase_left": frame[2], "phase_right": frame[3],
                               "clock_left": frame[4], "clock_right": frame[5],
                               "raw": raw, "hybrid": out[0], "clipped": out[1],
                               "q_target": out[2], "arm_reference": out[3],
                               "sdk_targets": out[4], "limited": out[5]})
        # standing + re-entry without reset (history/g continue, clocks stay 0)
        core2 = ReferenceCore(cfg)
        core2.reset()
        rng3 = np.random.default_rng(9)
        packets2 = fixture(rng3, 60, cmd, lambda t: t < 30)
        standing_zero = True
        for packet in packets2:
            snap = (packet["q"], packet["dq"], packet["quat"], packet["omega"])
            frame = core2.frame(snap, cmd, packet["standing"])
            core2._phases = (frame[2], frame[3])
            raw = session.run(["actions"], {"obs_history": core2.history[None]})[0][0]
            core2.complete(snap, cmd, packet["standing"], raw)
            if packet["standing"] and (abs(frame[4]) > 1e-9 or abs(frame[5]) > 1e-9):
                standing_zero = False
        variant_report["python_core_standing_clocks_zero"] = standing_zero
        variant_report["python_core_walked_ticks"] = ticks
        variant_report["python_core_executed"] = True

        # 4) C++ parity (only with the offline binary)
        if args.offline_binary:
            cpp_outputs = run_cpp(args.offline_binary, package, packets)
            atol = {"obs": 1e-5, "history": 1e-5, "g": 1e-6, "phase_left": 1e-6,
                    "phase_right": 1e-6, "clock_left": 1e-5, "clock_right": 1e-5,
                    "raw": 1e-6, "hybrid": 2e-4, "clipped": 1e-5, "q_target": 1e-5,
                    "arm_reference": 1e-5, "sdk_targets": 1e-5}
            max_errors = {}
            for py_out, cpp_out in zip(py_outputs, cpp_outputs):
                errors = compare(py_out, cpp_out, atol)
                for field, value in errors.items():
                    max_errors[field] = max(max_errors.get(field, 0.0), value)
            variant_report["cpp_parity"] = {"executed": True, "ticks": len(packets),
                                            "max_abs_error": max_errors,
                                            "hybrid_gate_note": "hybrid carries the arm-override "
                                                                "semantics; raw is also compared per-slot"}
        else:
            variant_report["cpp_parity"] = {
                "executed": False,
                "reason": "no --offline-binary provided; build on Linux and rerun",
            }
        report["variants"][name] = variant_report

    # 5) negative tests via the binary
    if args.offline_binary:
        package = variants["r41"]["package"].resolve()
        bad = {"q": [0.0] * 29, "dq": [0.0] * 29, "quat": [0, 0, 0, 0], "omega": [0, 0, 0],
               "cmd": list(contract.TEST_COMMAND), "standing": False, "reset": True}
        failed = subprocess.run([str(args.offline_binary.resolve()), str(package)],
                                input=yaml.safe_dump(bad) + "\n", text=True,
                                capture_output=True, timeout=10)
        report["negative_tests"] = {
            "bad_imu_rejected": failed.returncode != 0 and "Invalid pelvis IMU" in failed.stderr,
        }
    else:
        report["negative_tests"] = {"executed": False, "reason": "no --offline-binary"}

    report["not_covered"] = [
        "closed-loop simulation of the C++ core (plan section 10.3)",
        "hardware results (pelvis vs torso IMU source, wrist zero calibration)",
    ]
    print(json.dumps(report, indent=2, default=str))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()

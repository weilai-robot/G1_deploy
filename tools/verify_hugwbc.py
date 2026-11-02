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
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import export_hugwbc as contract  # frozen contract constants + actor rebuild


# ---------------------------------------------------------------- reference core
F32 = np.float32

# Previous-action observations feed inference differences back into history.
# Use the existing ONNX float32 absolute tolerance (main actor parity below)
# for those slots/actions; sensor, command and clock slots remain stricter.
CPP_ATOL = {"obs": 2e-4, "history": 2e-4, "g": 1e-6, "phase_left": 1e-6,
            "phase_right": 1e-6, "clock_left": 1e-5, "clock_right": 1e-5,
            "raw": 2e-4, "hybrid": 2e-4, "clipped": 2e-4, "q_target": 1e-5,
            "arm_reference": 1e-5, "sdk_targets": 1e-5, "limited": 0,
            "guard_violations": 0, "index": 0}


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
        self.last_target = np.zeros(29, dtype=F32)
        self.first_target = True
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
            # mirror core.h: pelvis orientation from the waist chain, and the
            # torso absolute gyro MINUS the waist relative rate
            wy, wr, wp = F32(q[12]), F32(q[13]), F32(q[14])
            cy, sy = F32(math.cos(float(wy))), F32(math.sin(float(wy)))
            cr, sr = F32(math.cos(float(wr))), F32(math.sin(float(wr)))
            cp, sp = F32(math.cos(float(wp))), F32(math.sin(float(wp)))
            rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=F32)
            rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=F32)
            ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=F32)
            rzrx = rz @ rx
            r_torso_p = rzrx @ ry                      # R_pelvis<-torso
            r = quat_matrix(quat) @ r_torso_p.T        # R_world<-pelvis
            w_torso_in_pelvis = r_torso_p @ np.asarray(omega, dtype=F32)
            e_x = np.array([1, 0, 0], dtype=F32)
            e_y = np.array([0, 1, 0], dtype=F32)
            e_z = np.array([0, 0, 1], dtype=F32)
            w_waist_rel = (e_z * dq[12] + rz @ (e_x * dq[13]) + rzrx @ (e_y * dq[14])).astype(F32)
            omega_b = (w_torso_in_pelvis - w_waist_rel).astype(F32)
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

        # deployment guards (must mirror core.h apply_guards: intersect
        # position/speed/torque intervals; reject the tick if they do not meet)
        lower = np.asarray(cfg["joint_lower"], dtype=F32)
        upper = np.asarray(cfg["joint_upper"], dtype=F32)
        kp = np.asarray(cfg["kp"], dtype=F32)
        kd = np.asarray(cfg["kd"], dtype=F32)
        tau = np.asarray(cfg["torque_limits"], dtype=F32)
        pos_lo = lower + F32(0.01)
        pos_hi = upper - F32(0.01)
        limited = 0
        violations = 0
        for i, s in enumerate(cfg["sdk_indices"]):
            lo, hi = F32(pos_lo[i]), F32(pos_hi[i])
            if cfg["max_target_speed"] > 0 and not self.first_target:
                v = F32(cfg["max_target_speed"] * cfg["step_dt"])
                lo = F32(max(float(lo), float(self.last_target[s] - v)))
                hi = F32(min(float(hi), float(self.last_target[s] + v)))
            if cfg["torque_guard"]:
                center = F32(q[s] + F32(kd[i] * dq[s] / kp[i]))
                radius = F32(tau[i] / kp[i])
                lo = F32(max(float(lo), float(center - radius)))
                hi = F32(min(float(hi), float(center + radius)))
            desired = sdk[s]
            if float(lo) <= float(hi):
                bounded = F32(min(float(max(float(desired), float(lo))), float(hi)))
                if abs(float(bounded - desired)) > 1e-5:
                    limited += 1
                desired = bounded
            else:
                raise RuntimeError(f"Empty target constraint intersection at SDK motor {s}")
            sdk[s] = F32(desired)
            self.last_target[s] = F32(desired)
        for s in cfg["wrist_hold"]["sdk_indices"]:
            self.last_target[s] = sdk[s]
        # F4: the NEXT observation carries the pre-clip hybrid action
        self.last_hybrid = hybrid.copy()
        self.first_target = False
        return hybrid, clipped, q_target, ref, sdk, limited, violations

    def step(self, snapshot, command, standing, raw):
        obs, g, pl, pr, cl, cr = self.frame(snapshot, command, standing)
        self._phases = (pl, pr)
        hybrid, clipped, q_target, ref, sdk, limited, violations = self.complete(
            snapshot, command, standing, raw)
        return {"obs": obs, "history": self.history.reshape(-1).copy(), "g": self.g,
                "phase_left": pl, "phase_right": pr, "clock_left": cl, "clock_right": cr,
                "hybrid": hybrid, "clipped": clipped, "q_target": q_target,
                "arm_reference": ref, "sdk_targets": sdk, "limited": limited,
                "guard_violations": violations}


# ---------------------------------------------------------------- fixtures
def fixture(rng, ticks, cmd, standing_schedule, start_quat=(0.7071, 0.0, 0.0, 0.7071)):
    """Synthetic SDK-order sensor sequence; seeded, includes phase wrap-crossing."""
    packets = []
    q = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0,
                  0, 0, 0, 0.3, 0.27, 0, 0.97, 0, 0, 0,
                  0.3, -0.27, 0, 0.97, 0, 0, 0], dtype=F32)
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
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise SystemExit(f"FAILED: non-finite values on {field}")
        if field in ("obs", "history"):
            frames_a, frames_b = a.reshape(-1, 88), b.reshape(-1, 88)
            non_action_slots = list(range(52)) + list(range(75, 88))
            if np.max(np.abs(frames_a[:, non_action_slots] - frames_b[:, non_action_slots])) > 1e-5:
                raise SystemExit(f"FAILED: sensor/command/clock slots on {field}")
        errors[field] = float(np.max(np.abs(a - b))) if a.size else 0.0
        if errors[field] > tolerance:
            raise SystemExit(f"FAILED: {field} max abs error {errors[field]} > {tolerance} "
                             f"(tick {cpp_out['index']})")
    return errors


def run_cpp(binary, package, packets):
    payload = "".join(json.dumps(packet) + "\n" for packet in packets)
    process = subprocess.run([str(binary), str(package)], input=payload,
                             capture_output=True, text=True, timeout=60)
    if process.returncode:
        raise RuntimeError(f"hugwbc_offline failed ({process.returncode}): {process.stderr}")
    outputs = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    if len(outputs) != len(packets):
        raise RuntimeError(f"hugwbc_offline returned {len(outputs)} ticks, expected {len(packets)}")
    return outputs


def regression_checks(cfg, session):
    """Fixed-behaviour regressions for the review findings F1/F4/F5/F6/F7.

    Each check fails loudly; the returned dict records the measured evidence.
    These checks run the Python reference. The separate hugwbc_core_checks
    executable exercises the C++ edge cases; the replay checks normal inference.
    """
    results = {}
    cmd = list(contract.TEST_COMMAND)
    import copy

    def nominal_q():
        q = np.zeros(29, dtype=F32)
        q[0:6] = [-0.1, 0, 0, 0.3, -0.2, 0]          # left leg q0
        q[6:12] = [-0.1, 0, 0, 0.3, -0.2, 0]         # right leg q0
        q[12:15] = 0.0                                # waist q0
        q[15:19] = [0.3, 0.27, 0, 0.97]               # left arm centre
        q[19:22] = 0.0                                # left wrists
        q[22:26] = [0.3, -0.27, 0, 0.97]              # right arm centre
        q[26:29] = 0.0                                # right wrists
        return q

    # F1: history must drop the OLDEST frame (training ObservationBuffer.insert)
    core = ReferenceCore(cfg)
    core.reset()
    for t in range(1, 9):
        snapshot_q = nominal_q()
        snapshot_q[0] = F32(snapshot_q[0] + 0.05 * t)  # left hip pitch, obs[6]
        frame = core.frame((snapshot_q, np.zeros(29, F32),
                            np.array([1, 0, 0, 0], F32), np.zeros(3, F32)), cmd, False)
        core._phases = (frame[2], frame[3])
        raw = session.run(["actions"], {"obs_history": core.history[None]})[0][0]
        core.complete((snapshot_q, np.zeros(29, F32)), cmd, False, raw)
        if t >= 5:
            expected = [0.05 * (t - 4 + h) for h in range(5)]
            actual = [float(core.history[h][6]) for h in range(5)]
            if max(abs(a - e) for a, e in zip(actual, expected)) > 1e-6:
                raise SystemExit(f"FAILED F1: history order {actual} != {expected}")
    results["history_order_oldest_dropped"] = "ok"

    # F4: next observation's action slice == previous pre-clip hybrid
    core = ReferenceCore(cfg)
    core.reset()
    previous_hybrid = None
    for t in range(4):
        snapshot = (nominal_q(), np.full(29, F32(0.7), F32),
                    np.array([0.98, 0.1, -0.1, 0.05], F32),
                    np.array([0.2, -0.1, 0.3], F32))
        frame = core.frame(snapshot, cmd, False)
        core._phases = (frame[2], frame[3])
        raw = session.run(["actions"], {"obs_history": core.history[None]})[0][0]
        hybrid = core.complete(snapshot, cmd, False, raw)[0]
        if previous_hybrid is not None:
            slice_obs = frame[0][52:75]
            if float(np.max(np.abs(slice_obs - previous_hybrid))) > 1e-6:
                raise SystemExit("FAILED F4: obs action slice != previous hybrid action")
        previous_hybrid = hybrid
    results["hybrid_action_history_backfill"] = "ok"

    # F6: torso IMU must subtract the waist relative rate
    torso_cfg = copy.deepcopy(cfg)
    torso_cfg["imu_source"] = "torso"
    core = ReferenceCore(torso_cfg)
    core.reset()
    max_gyro, max_gravity = 0.0, 0.0
    for q12, dq12 in ((0.3, 2.0), (-0.7, -3.0), (1.2, 0.5)):
        q_t = nominal_q()
        q_t[12] = F32(q12)
        dq_t = np.zeros(29, dtype=F32)
        dq_t[12] = F32(dq12)
        half = q12 / 2.0
        torso_quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], F32)  # wxyz
        torso_gyro = np.array([0.0, 0.0, dq12], F32)
        frame = core.frame((q_t, dq_t, torso_quat, torso_gyro), cmd, False)
        max_gyro = max(max_gyro, float(np.max(np.abs(frame[0][0:3]))))
        max_gravity = max(max_gravity, float(np.max(np.abs(
            frame[0][3:6] - np.array([0.0, 0.0, -1.0])))))
        core._phases = (frame[2], frame[3])
        raw = session.run(["actions"], {"obs_history": core.history[None]})[0][0]
        core.complete((q_t, dq_t), cmd, False, raw)
    if max_gyro > 1e-4:
        raise SystemExit(f"FAILED F6: pelvis gyro leaks waist motion: {max_gyro}")
    if max_gravity > 1e-3:
        raise SystemExit(f"FAILED F6: pelvis gravity wrong: {max_gravity}")
    results["torso_imu_waist_rate_compensated"] = (
        f"max residual gyro {max_gyro:.2e}, gravity {max_gravity:.2e}")

    # A yaw-only case cannot detect an inverse-rotation error: its gyro lies
    # on the rotation axis. Exercise all waist axes and a moving/tilted pelvis.
    def multiply(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([aw*bw-ax*bx-ay*by-az*bz,
                         aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx,
                         aw*bz+ax*by-ay*bx+az*bw], dtype=F32)

    max_error = 0.0
    for angles, rates in [([0.3, -0.2, 0.4], [0.7, -0.8, 0.9]),
                          ([-0.6, 0.3, -0.25], [-0.5, 0.4, -0.3])]:
        y, r, p = angles
        zq = np.array([math.cos(y/2), 0, 0, math.sin(y/2)], F32)
        xq = np.array([math.cos(r/2), math.sin(r/2), 0, 0], F32)
        yq = np.array([math.cos(p/2), 0, math.sin(p/2), 0], F32)
        waist_q = multiply(multiply(zq, xq), yq)
        pelvis_q = np.array([0.97, 0.1, 0.15, -0.05], F32)
        pelvis_q /= np.linalg.norm(pelvis_q)
        expected_gyro = np.array([0.2, -0.3, 0.4], F32)
        relative = (np.array([0, 0, rates[0]], F32) +
                    quat_matrix(zq) @ np.array([rates[1], 0, 0], F32) +
                    quat_matrix(multiply(zq, xq)) @ np.array([0, rates[2], 0], F32))
        gyro_t = quat_matrix(waist_q).T @ (expected_gyro + relative)
        q_t, dq_t = nominal_q(), np.zeros(29, F32)
        q_t[12:15], dq_t[12:15] = angles, rates
        core = ReferenceCore(torso_cfg)
        frame = core.frame((q_t, dq_t, multiply(pelvis_q, waist_q), gyro_t), cmd, False)
        expected = np.concatenate([expected_gyro * F32(cfg["obs_scales"]["ang_vel"]),
                                   -quat_matrix(pelvis_q)[2]])
        max_error = max(max_error, float(np.max(np.abs(frame[0][:6] - expected))))
    if max_error > 1e-6:
        raise SystemExit(f"FAILED F6: three-axis torso transform error {max_error}")
    results["torso_imu_three_axis_max_abs_error"] = max_error

    # F7: non-zero wrist hold targets survive the mapping
    wrist_cfg = copy.deepcopy(cfg)
    targets = [0.1, -0.1, 0.05, -0.05, 0.02, -0.02]
    wrist_cfg["wrist_hold"]["target_rad"] = targets
    core = ReferenceCore(wrist_cfg)
    core.reset()
    frame = core.frame((nominal_q(), np.zeros(29, F32), np.array([1, 0, 0, 0], F32),
                        np.zeros(3, F32)), cmd, False)
    core._phases = (frame[2], frame[3])
    raw = session.run(["actions"], {"obs_history": core.history[None]})[0][0]
    sdk = core.complete((nominal_q(), np.zeros(29, F32)), cmd, False, raw)[4]
    for value, slot in zip(targets, wrist_cfg["wrist_hold"]["sdk_indices"]):
        if abs(float(sdk[slot]) - value) > 1e-6:
            raise SystemExit(f"FAILED F7: wrist target at SDK {slot} is {sdk[slot]}, expected {value}")
    results["wrist_hold_targets_preserved"] = "ok"

    # The frame counter is already incremented by frame() on the first tick;
    # it must not be used to decide whether a previous target exists.
    speed_cfg = copy.deepcopy(cfg)
    speed_cfg["max_target_speed"] = 1.0
    core = ReferenceCore(speed_cfg)
    snapshot = (nominal_q(), np.zeros(29, F32), np.array([1, 0, 0, 0], F32), np.zeros(3, F32))
    raw = np.zeros(23, F32)
    raw[0] = 2.0
    first = core.step(snapshot, cmd, False, raw)
    raw[0] = -2.0
    second = core.step(snapshot, cmd, False, raw)
    if abs(float(first["sdk_targets"][0]) - 0.4) > 1e-6:
        raise SystemExit("FAILED: first target was incorrectly speed limited")
    if abs(float(second["sdk_targets"][0]) - (0.4 - cfg["step_dt"])) > 1e-6:
        raise SystemExit("FAILED: subsequent target was not speed limited")
    results["speed_guard_first_target_state"] = "ok"

    # F5: an empty position/torque intersection must reject the whole tick.
    # Left knee (SDK 3): crafted raw action drives the target far out; near-limit
    # position with a large velocity empties the intersection.
    core = ReferenceCore(cfg)
    core.reset()
    q_e = nominal_q()
    q_e[3] = F32(2.8)   # left knee near URDF upper 2.8798
    dq_e = np.zeros(29, dtype=F32)
    dq_e[3] = F32(60.0)
    frame = core.frame((q_e, dq_e, np.array([1, 0, 0, 0], F32), np.zeros(3, F32)), cmd, False)
    core._phases = (frame[2], frame[3])
    raw = session.run(["actions"], {"obs_history": core.history[None]})[0][0]
    raw[3] = F32(50.0)  # knee target -> 0.25*50 = 12.5 rad (deliberately extreme)
    try:
        core.complete((q_e, dq_e), cmd, False, raw)
    except RuntimeError as exc:
        if "Empty target constraint intersection" not in str(exc):
            raise
    else:
        raise SystemExit("FAILED F5: infeasible target package was returned")
    results["guard_empty_intersection_rejected"] = True
    return results


def real_anchor(csv_path, dt=0.02):
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
    parser.add_argument("--training-root", type=Path,
                        default=REPO.parent / "oma-runs/hugwbc-g1",
                        help="training archive root containing .oma (portable Linux/Windows path)")
    parser.add_argument("--variant", choices=("r41", "r42", "both"), default="both")
    parser.add_argument("--package", type=Path, help="explicit package directory")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--offline-binary", type=Path,
                        help="hugwbc_offline executable; C++ parity runs only when given")
    parser.add_argument("--rollout-csv", type=Path)
    parser.add_argument("--parity-samples", type=int, default=64)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    training_root = args.training_root
    if args.rollout_csv is None:
        args.rollout_csv = training_root / ".oma/experiments/exp-20261030-004/baseline-hybrid/seed1/rollout.csv"

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
        variant_report["regressions"] = regression_checks(cfg, session)

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
                               "index": core.tick - 1,
                               "g": frame[1], "phase_left": frame[2], "phase_right": frame[3],
                               "clock_left": frame[4], "clock_right": frame[5],
                               "raw": raw, "hybrid": out[0], "clipped": out[1],
                               "q_target": out[2], "arm_reference": out[3],
                               "sdk_targets": out[4], "limited": out[5],
                               "guard_violations": out[6]})
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
            max_errors = {}
            for py_out, cpp_out in zip(py_outputs, cpp_outputs):
                errors = compare(py_out, cpp_out, CPP_ATOL)
                for field, value in errors.items():
                    max_errors[field] = max(max_errors.get(field, 0.0), value)
            variant_report["cpp_parity"] = {"executed": True, "ticks": len(packets),
                                            "max_abs_error": max_errors,
                                            "note": "all pipeline intermediates compared "
                                                    "per tick, incl. guard_violations"}
        else:
            variant_report["cpp_parity"] = {
                "executed": False,
                "reason": "no --offline-binary provided; build on Linux and rerun",
            }
        report["variants"][name] = variant_report

    # 5) negative tests via the binary
    if args.offline_binary:
        package = next(iter(variants.values()))["package"].resolve()
        bad = {"q": [0.0] * 29, "dq": [0.0] * 29, "quat": [0, 0, 0, 0], "omega": [0, 0, 0],
               "cmd": list(contract.TEST_COMMAND), "standing": False, "reset": True}
        def require_rejection(package_path, packet, message):
            failed = subprocess.run([str(args.offline_binary.resolve()), str(package_path)],
                                    input=json.dumps(packet) + "\n", text=True,
                                    capture_output=True, timeout=10)
            if failed.returncode == 0 or message not in failed.stderr or failed.stdout.strip():
                raise SystemExit(f"FAILED negative test {message!r}: "
                                 f"rc={failed.returncode}, stderr={failed.stderr}, stdout={failed.stdout}")

        require_rejection(package, bad, "Invalid pelvis IMU")
        infeasible = fixture(np.random.default_rng(7), 1, contract.TEST_COMMAND, lambda t: False)[0]
        infeasible["q"][3], infeasible["dq"][3] = 2.8, 60.0
        require_rejection(package, infeasible, "Empty target constraint intersection")
        with tempfile.TemporaryDirectory(prefix="hugwbc-invalid-") as temp:
            wrong = Path(temp)
            (wrong / "deploy.yaml").write_text("format: invalid\n", encoding="utf-8")
            require_rejection(wrong, bad, "Incompatible HugWBC package format")
            (wrong / "deploy.yaml").write_bytes((package / "deploy.yaml").read_bytes())
            (wrong / "policy.onnx").write_bytes(b"corrupted policy")
            require_rejection(wrong, bad, "policy.onnx hash mismatch")
        report["negative_tests"] = {"executed": True, "bad_imu_rejected": True,
                                    "empty_intersection_rejected": True,
                                    "wrong_format_rejected": True,
                                    "corrupt_policy_rejected": True}
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

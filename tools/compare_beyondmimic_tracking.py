#!/usr/bin/env python3
"""Record sim2sim tracking-error curves for BeyondMimic packages.

Training-replay condition (privileged state + raw targets, no deployment
clamps/odometry) — identical target generation to scripts/play_mujoco.py
(default + action*action_scale, effort-limited PD) and identical metric
definitions to the training command (tasks/tracking/mdp/commands.py):
error_body_pos is the 14-body mean position error against the reference
re-anchored exactly as _update_command does — yaw-only re-orientation,
horizontal position from the robot's torso, vertical pinned to the reference
anchor (delta_pos_w = robot torso xy + ref torso z; delta_ori_w = yaw-only).
"""
import argparse
import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
import yaml

INK = "#1a1a19"
MUTED = "#6b6a64"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834"]  # validated categorical slots 1-2 (light)


def matrix(q):
    return Rotation.from_quat(np.roll(q, -1)).as_matrix()


def yaw_of(r):
    c, s = r[0, 0], r[1, 0]
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def simulate(package, model, data, seconds):
    package = package.resolve()
    cfg = yaml.safe_load((package/"deploy.yaml").read_text())
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
    qids = np.array([model.jnt_qposadr[model.joint(n).id] for n in sdk_names])
    vids = np.array([model.jnt_dofadr[model.joint(n).id] for n in sdk_names])
    actuator_ids = np.array([next(i for i in range(model.nu)
                                  if model.actuator_trnid[i, 0] == model.joint(n).id) for n in sdk_names])
    torso = model.body("torso_link").id
    body_ids = np.array([model.body(n).id for n in cfg["body_names"]])
    defaults = np.asarray(cfg["default_joint_pos"])
    kp, kd, effort = np.zeros(29), np.zeros(29), np.zeros(29)
    kp[ids], kd[ids], effort[ids] = cfg["kp"], cfg["kd"], cfg["effort_limit"]
    action_scale = np.asarray(cfg["action_scale"])
    session = ort.InferenceSession(str(package/"policy.onnx"), providers=["CPUExecutionProvider"])

    def initialise():
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = [0, 0, ref_pos[0, 0, 2]]
        data.qpos[3:7] = ref_quat[0, 0]
        data.qpos[qids[ids]] = ref_q[0]
        mujoco.mj_forward(model, data)

    # Deployment aligns the motion's initial root yaw/origin to the robot (as the
    # C++ core's reset does); all world-frame comparisons use this aligned frame.
    initialise()
    rot = matrix(data.qpos[3:7])
    ref_rot = matrix(ref_quat[0, 0])
    alignment = Rotation.from_euler("z", np.arctan2(rot[1, 0], rot[0, 0])
                                    - np.arctan2(ref_rot[1, 0], ref_rot[0, 0])).as_matrix()
    translation = data.qpos[:3].copy() - alignment@ref_pos[0, 0]

    steps = min(int(seconds/dt), frames)
    substeps = round(dt/model.opt.timestep)
    last_action = np.zeros(29)
    curve = {"t": [], "error_body_pos_mean": [], "error_body_pos_max": [],
             "error_anchor_pos": [], "error_joint_pos": [], "pelvis_z": [], "tilt": []}
    trip_frame = None
    for frame in range(steps):
        torso_r = matrix(data.xquat[torso])
        obs = np.concatenate([
            ref_q[frame], ref_dq[frame],
            torso_r.T@(alignment@ref_pos[frame, 7]+translation-data.xpos[torso]),
            (torso_r.T@alignment@matrix(ref_quat[frame, 7]))[:, :2].reshape(-1),
            matrix(data.qpos[3:7]).T@data.qvel[:3], data.qvel[3:6],
            data.qpos[qids[ids]]-defaults, data.qvel[vids[ids]], last_action,
        ]).astype(np.float32)
        action = session.run(["actions"], {"obs": obs[None]})[0][0]
        last_action = np.asarray(action)
        targets = np.empty(29)
        targets[ids] = defaults + action_scale*np.asarray(action)
        for _ in range(substeps):
            tau = np.clip(kp*(targets-data.qpos[qids])-kd*data.qvel[vids], -effort, effort)
            data.ctrl[actuator_ids] = tau
            mujoco.mj_step(model, data)
        mujoco.mj_kinematics(model, data)

        # Training metric (commands.py _update_command): yaw-only re-orientation,
        # horizontal from robot torso, vertical pinned to the reference anchor.
        torso_r = matrix(data.xquat[torso])
        ref_world = ref_pos[frame]@alignment.T + translation
        delta_ori = yaw_of(torso_r)@yaw_of(alignment@matrix(ref_quat[frame, 7])).T
        delta_pos = np.array([data.xpos[torso][0], data.xpos[torso][1], ref_world[7][2]])
        rel_targets = delta_pos + (delta_ori@(ref_world-ref_world[7]).T).T
        body_err = np.linalg.norm(data.xpos[body_ids]-rel_targets, axis=1)
        tilt = float(np.arccos(np.clip(torso_r[2, 2], -1, 1)))
        curve["t"].append((frame+1)*dt)
        curve["error_body_pos_mean"].append(float(body_err.mean()))
        curve["error_body_pos_max"].append(float(body_err.max()))
        curve["error_anchor_pos"].append(float(np.linalg.norm(data.xpos[torso]-ref_world[7])))
        curve["error_joint_pos"].append(float(np.linalg.norm(data.qpos[qids[ids]]-ref_q[frame])))
        curve["pelvis_z"].append(float(data.qpos[2]))
        curve["tilt"].append(tilt)
        if trip_frame is None and (data.qpos[2] < 0.35 or tilt > 0.85):
            trip_frame = frame
        if not np.isfinite(data.qpos).all():
            break

    windows = {}
    for lo, hi in ((0, 10), (10, 20), (20, 30)):
        vals = [v for t, v in zip(curve["t"], curve["error_body_pos_mean"]) if lo <= t < hi]
        if vals:
            windows[f"{lo}-{hi}s"] = {"mean": round(float(np.mean(vals)), 4),
                                      "max": round(float(np.max(vals)), 4)}
    summary = {"package": package.name, "condition": "truth state + raw targets (training replay)",
               "frames_recorded": len(curve["t"]), "simulated_seconds": round(len(curve["t"])*dt, 2),
               "abort_threshold_tripped_at_frame": trip_frame,
               "abort_threshold_tripped_at_s": None if trip_frame is None else round((trip_frame+1)*dt, 2),
               "error_body_pos_overall_mean": round(float(np.mean(curve["error_body_pos_mean"])), 4),
               "error_body_pos_overall_max": round(float(np.max(curve["error_body_pos_mean"])), 4),
               "windows_error_body_pos_mean": windows}
    return curve, summary


def plot(curves, labels, out_png, seconds):
    plt.rcParams.update({"font.size": 10})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.8), dpi=150, sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=MUTED)
    for curve, label, color in zip(curves, labels, SERIES):
        t = curve["t"]
        ax1.plot(t, curve["error_body_pos_mean"], color=color, linewidth=2, label=label)
        ax2.plot(t, curve["error_anchor_pos"], color=color, linewidth=2)
        trip = next((i for i, (z, tilt) in enumerate(zip(curve["pelvis_z"], curve["tilt"]))
                     if z < 0.35 or tilt > 0.85), None)
        if trip is not None:
            for ax in (ax1, ax2):
                ax.axvline(t[trip], color=color, linewidth=1, linestyle=(0, (4, 3)), alpha=0.75)
            ax1.annotate(f"{label} abort tripped {t[trip]:.1f}s", (t[trip], ax1.get_ylim()[1]),
                         xytext=(t[trip]+0.3, None), textcoords=("data", "axes fraction"),
                         color=MUTED, fontsize=8, va="top")
    ax1.set_ylabel("error_body_pos mean (m)", color=INK)
    ax2.set_ylabel("torso anchor error (m)", color=INK)
    ax2.set_xlabel("time (s)", color=INK)
    ax2.axhline(0.35, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
    ax2.annotate("deployment max_anchor_error 0.35 m", (0.995, 0.35), xycoords=("axes fraction", "data"),
                 color=MUTED, fontsize=8, va="bottom", ha="right")
    ax1.legend(frameon=False, loc="upper right", labelcolor=INK)
    fig.suptitle("BeyondMimic sim2sim tracking replay, first 30 s", color=INK,
                 fontsize=13, x=0.5, y=0.98)
    ax1.set_title("privileged state + raw targets (play_mujoco-equivalent); no deployment clamps/odometry",
                  color=MUTED, fontsize=9, loc="left", pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packages", type=Path, nargs="+", required=True)
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    labels = args.labels if args.labels else [pkg.name for pkg in args.packages]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    curves, summaries = [], []
    for package, label in zip(args.packages, labels):
        curve, summary = simulate(package, model, data, args.seconds)
        summary["label"] = label
        curves.append(curve)
        summaries.append(summary)
        (out/f"{label}_curves.json").write_text(json.dumps(curve)+"\n")
        print(json.dumps(summary, indent=2))
    plot(curves, labels, out/"error_curves.png", args.seconds)
    (out/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")


if __name__ == "__main__":
    main()

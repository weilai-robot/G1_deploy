#!/usr/bin/env python3
"""S1 grounding repair for the gLO (guohua_show_dance) reference motion.

GVHMR video mocap drifts in world vertical: the whole body floats 8-18 cm
above its contact plane for 46.6% of frames (longest 7.46 s at standing
pelvis height — physically impossible). Relative pose (joint angles, body
configuration) is trustworthy; only world-frame root translation z drifts.

Repair (analysis doc §七 S1):
1. Estimate the slow artifact a(t) from the lowest foot: sliding low
   percentile anchored at the contact plane, smoothed, then clamped so the
   repaired feet never sink more than eps below the plane (guards against
   over-subtraction at genuine fast foot touches).
2. Subtract a(t) rigidly from every body's world z and from the z of every
   body's linear velocity — the training reward tracks body lin/ang vel
   (rewards.py:61,72), so velocities must stay consistent with positions.
3. Splice a FixStand → first-frame transition (cubic Hermite with
   end-velocity matching, MuJoCo FK for all 30 bodies) so the FSM start
   check and RSI get an in-distribution entry; gLO has 0/2400 frames
   passing the 0.25 rad start check.

Writes a NEW npz next to the original; the original file is never modified.
"""
import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.spatial.transform import Rotation

INK, MUTED, GRID, AXIS, SURFACE = "#1a1a19", "#6b6a64", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834"]  # validated categorical slots 1-2 (light)

# .npz stores all 30 bodies in PhysX BFS order (play_mujoco.py NPZ_BODY_ORDER).
NPZ_BODY_ORDER = [
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]
FOOT = (NPZ_BODY_ORDER.index("left_ankle_roll_link"),
        NPZ_BODY_ORDER.index("right_ankle_roll_link"))


def assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise ValueError(f"Missing assignment: {name}")


def wxyz(quat):
    """npz quats are wxyz; scipy wants xyzw."""
    return np.roll(quat, -1, axis=-1)


def longest_run(mask):
    best = cur = 0
    for flag in mask:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def hermite(p0, v0, p1, v1, u, T):
    """Cubic Hermite P(u) and dP/dt, elementwise over arrays; endpoint
    velocities matched exactly (v0, v1 in units/second)."""
    u2, u3 = u * u, u * u * u
    h00, h10, h01, h11 = 2*u3-3*u2+1, u3-2*u2+u, -2*u3+3*u2, u3-u2
    d00, d10, d01, d11 = 6*u2-6*u, 3*u2-4*u+1, -6*u2+6*u, 3*u2-2*u
    P = h00*p0 + h10*T*v0 + h01*p1 + h11*T*v1
    dPdt = (d00*p0 + d10*T*v0 + d01*p1 + d11*T*v1) / T
    return P, dPdt


def quintic(u):
    return 10*u**3 - 15*u**4 + 6*u**5


def slerp_quat(q0w, q1w, t):
    r0 = Rotation.from_quat(wxyz(q0w))
    r1 = Rotation.from_quat(wxyz(q1w))
    rel = r0.inv() * r1
    return np.roll((r0 * Rotation.from_rotvec(rel.as_rotvec() * t)).as_quat(), 1)


def foot_z(motion):
    return np.minimum(motion["body_pos_w"][:, FOOT[0], 2],
                      motion["body_pos_w"][:, FOOT[1], 2])


def estimate_artifact(foot, plane, fps, args):
    """Slow vertical artifact a(t) anchored at the contact plane, with the
    projection clamp so repaired feet never sink >eps below the plane."""
    w = int(round(args.window_s * fps)) // 2 * 2 + 1
    a = gaussian_filter1d(percentile_filter(foot, args.pct, size=w,
                                            mode="nearest"), args.smooth_s * fps) - plane
    a = np.clip(a, -0.005, None)
    for sigma in [args.smooth_s * fps / 3] * 4:
        a = np.minimum(a, foot - plane + args.eps)   # anti over-subtraction
        a = gaussian_filter1d(a, sigma)
    a = np.minimum(a, foot - plane + args.eps)
    return a


def stats(foot, plane, body_min_z, pelvis_z, dt):
    rel = foot - plane
    return {
        "grounded_frac_foot_lt2cm": round(float((rel < 0.02).mean()), 4),
        "foot_above10cm_frac": round(float((rel > 0.10).mean()), 4),
        "longest_foot_above10cm_s": round(longest_run(rel > 0.10) * dt, 2),
        "longest_no_body_support_s": round(longest_run(body_min_z - plane > 0.05) * dt, 2),
        "median_foot_above_plane_m": round(float(np.median(rel)), 4),
        "min_foot_above_plane_m": round(float(rel.min()), 4),
        "pelvis_z_range_m": [round(float(pelvis_z.min()), 3), round(float(pelvis_z.max()), 3)],
    }


def pick_velocity_convention(pos, quat, lin, ang, dt):
    """Match the npz velocity arrays against finite-difference estimators on
    the original frames; return (lin_of, ang_of, report). ang_of/lin_of take
    the sequence they should difference, so they work on any timeline."""
    k = np.arange(1, len(pos) - 1)
    lin_c = (pos[k+1] - pos[k-1]) / (2*dt)
    lin_f = (pos[k+1] - pos[k]) / dt
    T_n, B, _ = quat.shape
    Rb = Rotation.from_quat(wxyz(quat).reshape(-1, 4))     # flat (T*B,)
    n = len(k)
    kf = ((k * B)[:, None] + np.arange(B)[None, :]).ravel()   # flat idx of frame k

    def rv(rel):
        return rel.as_rotvec().reshape(n, B, 3)

    ang_w_f = rv(Rb[kf + B] * Rb[kf].inv()) / dt                    # world frame
    ang_w_c = rv(Rb[kf + B] * Rb[kf - B].inv()) / (2*dt)
    ang_b_f = rv(Rb[kf].inv() * Rb[kf + B]) / dt                    # body frame
    ang_b_c = rv(Rb[kf - B].inv() * Rb[kf + B]) / (2*dt)
    report = {
        "lin_central_mps": round(float(np.abs(lin_c - lin[k]).mean()), 5),
        "lin_forward_mps": round(float(np.abs(lin_f - lin[k]).mean()), 5),
        "ang_world_forward_rads": round(float(np.abs(ang_w_f - ang[k]).mean()), 5),
        "ang_world_central_rads": round(float(np.abs(ang_w_c - ang[k]).mean()), 5),
        "ang_body_forward_rads": round(float(np.abs(ang_b_f - ang[k]).mean()), 5),
        "ang_body_central_rads": round(float(np.abs(ang_b_c - ang[k]).mean()), 5),
    }
    lin_is_central = report["lin_central_mps"] <= report["lin_forward_mps"]

    def lin_of(seq, j):
        if j + 1 >= len(seq):
            return (seq[j] - seq[j-1]) / dt
        if lin_is_central and j > 0:
            return (seq[j+1] - seq[j-1]) / (2*dt)
        return (seq[j+1] - seq[j]) / dt

    ang_name = min(("ang_world_forward_rads", "ang_world_central_rads",
                    "ang_body_forward_rads", "ang_body_central_rads"),
                   key=report.get)
    world = "world" in ang_name
    central = "central" in ang_name

    def ang_of(seq, j):
        q = Rotation.from_quat(wxyz(seq))
        if j + 1 >= len(q):
            rel = (q[j] * q[j-1].inv()) if world else (q[j-1].inv() * q[j])
            return rel.as_rotvec() / dt
        if central and j > 0:
            rel = (q[j+1] * q[j-1].inv()) if world else (q[j-1].inv() * q[j+1])
            return rel.as_rotvec() / (2*dt)
        rel = (q[j+1] * q[j].inv()) if world else (q[j].inv() * q[j+1])
        return rel.as_rotvec() / dt

    report["chosen"] = ["lin_" + ("central" if lin_is_central else "forward"), ang_name]
    return lin_of, ang_of, report


def fk_bodies(model, data, joint_adr, root_p, root_qw, joints):
    data.qpos[:3] = root_p
    data.qpos[3:7] = root_qw
    data.qpos[joint_adr] = joints
    mujoco.mj_kinematics(model, data)
    pos = np.stack([data.xpos[model.body(n).id] for n in NPZ_BODY_ORDER])
    quat = np.stack([data.xquat[model.body(n).id] for n in NPZ_BODY_ORDER])
    return pos, quat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--stand-motion", type=Path, required=True,
                   help="npz whose frame 0 is the FixStand anchor (dance_zui)")
    p.add_argument("--playback-script", type=Path, required=True,
                   help="play_mujoco.py, parsed for ISAAC_JOINT_ORDER")
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--window-s", type=float, default=2.5)
    p.add_argument("--pct", type=float, default=10.0)
    p.add_argument("--smooth-s", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--trans-s", type=float, default=1.0)
    args = p.parse_args()

    raw = dict(np.load(args.motion, allow_pickle=False))
    stand = np.load(args.stand_motion, allow_pickle=False)
    fps = float(np.asarray(raw["fps"]).flat[0])
    dt = 1.0 / fps
    foot = foot_z(raw)
    plane = float(np.percentile(foot, 5))

    a = estimate_artifact(foot, plane, fps, args)
    print(f"[plane] contact plane z0 = {plane:.4f} m; artifact median "
          f"{np.median(a):.3f} m, max {a.max():.3f} m")

    fixed = {k: v.copy() for k, v in raw.items()}
    fixed["body_pos_w"][:, :, 2] -= a[:, None]
    fixed["body_lin_vel_w"][:, :, 2] -= np.gradient(a, dt)[:, None]

    # ---- MuJoCo FK self-test: does our FK reproduce the npz bodies? ----
    isaac_order = ast.literal_eval(assignment(
        ast.parse(args.playback_script.read_text()), "ISAAC_JOINT_ORDER"))
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    data = mujoco.MjData(model)
    joint_adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in isaac_order])

    probe = [0, 100, 600, 1200, 2000]
    fk_err_p, fk_err_q = [], []
    for f in probe:
        pos, quat = fk_bodies(model, data, joint_adr,
                              fixed["body_pos_w"][f, 0], fixed["body_quat_w"][f, 0],
                              fixed["joint_pos"][f])
        fk_err_p.append(np.abs(pos - fixed["body_pos_w"][f]).max())
        fk_err_q.append(float((Rotation.from_quat(wxyz(quat))
                               * Rotation.from_quat(wxyz(fixed["body_quat_w"][f])).inv()
                               ).magnitude().max()))
    print(f"[fk-selftest] frames {probe}: max|dpos| = {max(fk_err_p)*1000:.2f} mm, "
          f"max|dangle| = {max(fk_err_q)*1000:.2f} mrad")
    if max(fk_err_p) > 0.008:
        raise SystemExit("FK self-test failed — skeleton/order mismatch, aborting splice")

    lin_of, ang_of, conv = pick_velocity_convention(
        raw["body_pos_w"], raw["body_quat_w"], raw["body_lin_vel_w"],
        raw["body_ang_vel_w"], dt)
    print(f"[vel-convention] {conv}")

    # ---- splice: FixStand → repaired first frame ----
    N = int(round(args.trans_s * fps))
    T = N * dt
    s_q = stand["joint_pos"][0].astype(np.float64)
    s_root_p = stand["body_pos_w"][0, 0].astype(np.float64)
    s_root_p[2] += plane - float(np.percentile(foot_z(stand), 5))
    s_root_q = stand["body_quat_w"][0, 0].astype(np.float64)
    g_q = fixed["joint_pos"][0].astype(np.float64)
    g_dq = fixed["joint_vel"][0].astype(np.float64)
    g_root_p = fixed["body_pos_w"][0, 0].astype(np.float64)
    g_root_v = fixed["body_lin_vel_w"][0, 0].astype(np.float64)
    g_root_q = fixed["body_quat_w"][0, 0].astype(np.float64)

    tr_j = np.zeros((N, 29)); tr_jv = np.zeros((N, 29))
    tr_p = np.zeros((N, 3)); tr_v = np.zeros((N, 3))
    tr_q = np.zeros((N, 4))
    for i, u in enumerate(np.arange(N) / N):
        tr_j[i], tr_jv[i] = hermite(s_q, 0.0, g_q, g_dq, u, T)
        tr_p[i], tr_v[i] = hermite(s_root_p, 0.0, g_root_p, g_root_v, u, T)
        tr_q[i] = slerp_quat(s_root_q, g_root_q, quintic(u))
    fk = [fk_bodies(model, data, joint_adr, tr_p[i], tr_q[i], tr_j[i]) for i in range(N)]
    tr_bodies = np.stack([f[0] for f in fk])
    tr_bquats = np.stack([f[1] for f in fk])

    # transition velocities on the concatenated timeline (self-consistent)
    cat_p = np.concatenate([tr_bodies, fixed["body_pos_w"]])
    cat_q = np.concatenate([tr_bquats, fixed["body_quat_w"]])
    tr_bv = np.zeros((N, 30, 3)); tr_bw = np.zeros((N, 30, 3))
    for i in range(1, N):
        for b in range(30):
            tr_bv[i, b] = lin_of(cat_p[:, b], i)
            tr_bw[i, b] = ang_of(cat_q[:, b], i)
    tr_bv[:, 0] = tr_v          # pelvis linear vel: analytic Hermite (exact endpoints)

    out = dict(raw)
    out["joint_pos"] = np.concatenate([tr_j, fixed["joint_pos"]]).astype(np.float32)
    out["joint_vel"] = np.concatenate([tr_jv, fixed["joint_vel"]]).astype(np.float32)
    out["body_pos_w"] = cat_p.astype(np.float32)
    out["body_quat_w"] = cat_q.astype(np.float32)
    out["body_lin_vel_w"] = np.concatenate([tr_bv, fixed["body_lin_vel_w"]]).astype(np.float32)
    out["body_ang_vel_w"] = np.concatenate([tr_bw, fixed["body_ang_vel_w"]]).astype(np.float32)

    flat = np.concatenate([v.ravel() for v in out.values() if v.dtype.kind == "f"])
    if not np.isfinite(flat).all():
        raise SystemExit("non-finite output")

    # ---- verification stats ----
    n_tr = len(tr_j)
    foot_new = foot_z(out)
    body_min_new = out["body_pos_w"][:, :, 2].min(axis=1)
    before = stats(foot, plane, raw["body_pos_w"][:, :, 2].min(axis=1),
                   raw["body_pos_w"][:, 0, 2], dt)
    after = stats(foot_new[n_tr:], plane, body_min_new[n_tr:],
                  out["body_pos_w"][n_tr:, 0, 2], dt)
    splice_after = stats(foot_new[:n_tr], plane, body_min_new[:n_tr],
                         out["body_pos_w"][:n_tr, 0, 2], dt)
    junction = {
        "start_check_frame0_max_rad": round(float(np.abs(out["joint_pos"][0] - s_q).max()), 6),
        "transition_span_rad": round(float(np.abs(g_q - s_q).max()), 3),
        "transition_joint_vel_max_rads": round(float(np.abs(out["joint_vel"][:n_tr]).max()), 3),
        "joint_vel_jump_max_rads": round(float(np.abs(
            out["joint_vel"][n_tr - 1] - out["joint_vel"][n_tr]).max()), 3),
        "body_vel_jump_max_mps": round(float(np.abs(
            out["body_lin_vel_w"][n_tr - 1] - out["body_lin_vel_w"][n_tr]).max()), 3),
    }
    summary = {
        "motion": args.motion.name, "output": args.out.name,
        "plane_m": round(plane, 4), "frames_before": len(foot),
        "frames_after": len(out["joint_pos"]), "transition_frames": n_tr,
        "artifact": {"median_m": round(float(np.median(a)), 3),
                     "max_m": round(float(a.max()), 3),
                     "drift_rate_max_mps": round(float(np.abs(np.gradient(a, dt)).max()), 4)},
        "before": before, "after": after, "transition_segment": splice_after,
        "junction": junction, "fk_selftest_mm": round(max(fk_err_p) * 1000, 2),
        "velocity_convention": conv,
    }
    print(json.dumps(summary, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(foot, foot_new[n_tr:], a + plane, plane, report / "grounding_curves.png",
         np.arange(len(foot)) * dt, np.arange(len(foot_new) - n_tr) * dt)


def plot(raw_foot, new_foot, artifact_track, plane, out_png, t_raw, t_new):
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=MUTED)
    ax.plot(t_raw, raw_foot, color=MUTED, linewidth=1.4, alpha=0.75, label="raw lowest foot")
    ax.plot(t_new, new_foot, color=SERIES[0], linewidth=2, label="repaired lowest foot")
    ax.plot(t_raw, artifact_track, color=SERIES[1], linewidth=2,
            label="artifact estimate a(t) (as absolute height)")
    ax.axhline(plane, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
    ax.annotate(f"contact plane {plane:.3f} m", (0.995, plane),
                xycoords=("axes fraction", "data"),
                color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("time (s)", color=INK)
    ax.set_ylabel("lowest foot height (m)", color=INK)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.set_title("gLO grounding repair (S1): slow vertical artifact removed",
                 color=INK, fontsize=12, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()

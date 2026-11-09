#!/usr/bin/env python3
"""MuJoCo HoloWbt FSM 模拟器 — 完整复现遥控器操作 Step6 流程。

运行方法:
    cd G1_deploy
    python tools/simulate_holo_wbt_fsm.py                    # 交互模式
    python tools/simulate_holo_wbt_fsm.py --record out.mp4   # 录制交互视频
    python tools/simulate_holo_wbt_fsm.py --auto --record demo.mp4  # 自动演示

键盘映射（模仿 SRC-6 遥控器）:
    LShift = L2      RCtrl = R1
    Up     = 方向上    Right = 方向右
    X, A, B = X A B 键
    Q     = 退出

操作序列（照 Step6）:
  1. LShift+Up   → Passive → FixStand
  2. 等几秒站稳
  3. LShift(hold)+Right → FixStand → Holo_Prepose
  4. RCtrl+X     → Holo_Prepose → Holo_Box (跑 1.78s)
  5. 自动完成     → Velocity
  6. RCtrl+X (箱内) → 提前结束 → Velocity
  7. RCtrl+A     → 任意状态 → Passive
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import yaml

# ------- package.h 的全 Python 等价实现 -------

J, B, O = 29, 14, 154
ANCHOR = 7  # torso_link index in body_names[7]


def read_bmg_v1(path: Path) -> tuple[int, float, np.ndarray]:
    raw = path.read_bytes()
    magic, frames, joints, bodies, dt = struct.unpack("<8sIIIf", raw[:24])
    assert magic == b"BMG1V1\0\0", f"bad magic {magic!r}"
    assert joints == J and bodies == B, f"header J={joints} B={bodies}"
    assert abs(dt - 0.02) < 1e-6, f"dt={dt}"
    recs = np.frombuffer(raw[24:], dtype=np.float32).reshape(frames, 2 * J + 7 * B)
    return frames, dt, recs


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    ang = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw, qx, qy, qz = 0.25*S, (m[2, 1]-m[1, 2])/S, (m[0, 2]-m[2, 0])/S, (m[1, 0]-m[0, 1])/S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw, qx, qy, qz = (m[2, 1]-m[1, 2])/S, 0.25*S, (m[0, 1]+m[1, 0])/S, (m[0, 2]+m[2, 0])/S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[2, 2] + m[0, 0]) * 2.0
        qw, qx, qy, qz = (m[0, 2]-m[2, 0])/S, (m[0, 1]+m[1, 0])/S, 0.25*S, (m[1, 2]+m[2, 1])/S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw, qx, qy, qz = (m[2, 0]-m[0, 2])/S, (m[0, 1]+m[1, 0])/S, (m[1, 2]+m[2, 1])/S, 0.25*S
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


def forward_kinematics(chain: list[dict], q: np.ndarray) -> np.ndarray:
    R = np.eye(3)
    for seg in chain:
        rpy = np.asarray(seg["rpy"], dtype=np.float64)
        cx, sx = np.cos(rpy[0]), np.sin(rpy[0])
        cy, sy = np.cos(rpy[1]), np.sin(rpy[1])
        cz, sz = np.cos(rpy[2]), np.sin(rpy[2])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = R @ (Rz @ Ry @ Rx)
        idx = seg["motor_index"]
        if idx >= 0:
            axis = np.asarray(seg["axis"], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            a = float(q[idx])
            x, y, z = axis
            c, s = np.cos(a), np.sin(a)
            Ra = np.array([
                [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
                [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)],
            ])
            R = R @ Ra
    return R


# ------- 状态常量 -------

(S_PASSIVE, S_FIXSTAND, S_HOLO_PREPOSE, S_HOLO_BOX, S_VELOCITY) = range(5)
STATE_NAMES = {
    S_PASSIVE: "PASSIVE",
    S_FIXSTAND: "FixStand",
    S_HOLO_PREPOSE: "Holo_Prepose",
    S_HOLO_BOX: "Holo_Box",
    S_VELOCITY: "Velocity",
}
STATE_COLORS = {
    S_PASSIVE: (100, 100, 100),
    S_FIXSTAND: (60, 180, 75),
    S_HOLO_PREPOSE: (60, 120, 220),
    S_HOLO_BOX: (60, 200, 220),
    S_VELOCITY: (180, 130, 60),
}


# ------- 主模拟器 -------

class HoloWbtFSMSim:
    def __init__(self, pkg_dir: str | Path, urdf: str | Path | None = None,
                 record: str | None = None, auto: bool = False, gantry: float = 0.0):
        import mujoco
        import onnxruntime as ort

        self.pkg = Path(pkg_dir)
        cfg = yaml.safe_load((self.pkg / "deploy.yaml").read_text("utf-8"))
        assert cfg["format"].startswith("holo_wbt_g1"), f"bad format: {cfg['format']}"
        self.cfg = cfg

        # 参考表
        self.ref_frames, self.ref_dt, self.ref_data = read_bmg_v1(self.pkg / "reference.bin")
        self.frame_count = self.ref_frames
        print(f"  reference: {self.ref_frames} frames @ {1.0/self.ref_dt:.0f} Hz")

        # ONNX
        self.ort = ort.InferenceSession(
            str(self.pkg / "policy.onnx"), providers=["CPUExecutionProvider"])
        inp, out = self.ort.get_inputs()[0], self.ort.get_outputs()[0]
        assert inp.name == "obs" and list(inp.shape) == [1, O]
        assert out.name == "actions" and list(out.shape) == [1, J]

        # 参数
        self.names = [str(n) for n in cfg["joint_names"]]
        self.kp = np.array(cfg["kp"], np.float64)
        self.kd = np.array(cfg["kd"], np.float64)
        self.scale = np.array(cfg["action_scale"], np.float64)
        self.limits = np.array(cfg["effort_limit"], np.float64)
        self.default = np.array(cfg["default_joint_pos"], np.float64)
        self.lo = np.array(cfg["joint_lower"], np.float64)
        self.hi = np.array(cfg["joint_upper"], np.float64)
        self.chain = cfg["kinematics"]["torso"]

        # 状态机
        self.state = S_FIXSTAND
        self.state_time = 0.0
        self.elapsed = 0.0
        self.last_action = np.zeros(J, np.float64)
        self.last_target = np.zeros(J, np.float64)
        self.alignment = np.eye(3)
        self.pose_start = np.zeros(J)
        self.pose_target = np.zeros(J)
        self.policy_ok = False
        self.clipped = 0
        self.fell = False
        self.running = True
        self._holo_tick = 0
        self._anchor_xy = None
        self._l2_hold_start = -1

        # 自动演示 / 录制
        self.auto = auto
        self.record_path = record
        self._writer = None
        self._fps = 25.0

        # 安全吊架: gantry>0 表示在骨盆施加向上力, 补偿总重×gantry
        self.gantry = gantry

        # --------- MuJoCo (完整 G1 模型, 含视觉网格) ---------
        scene_xml = Path(__file__).resolve().parents[1] / "../holosoma/src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml"
        scene_xml = scene_xml.resolve()
        if not scene_xml.exists():
            raise FileNotFoundError(f"MJCF scene not found: {scene_xml}")

        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.model.opt.timestep = 0.005
        self.model.opt.iterations = 50
        self.model.vis.global_.offwidth = 960
        self.model.vis.global_.offheight = 600
        self.data = mujoco.MjData(self.model)

        # 吊架: 安全绳式 骨盆高度低于阈值时施加拉力
        self._ref_height = 0.79  # standing torso_link height
        self._rope_slack = 0.00
        self._rope_k = 5000.0
        self._rope_d = 100.0
        self._rope_bodies = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link"),
        ]
        self._rope_ref_z = 0.0
        # 关节索引
        qid, vid, aid = [], [], []
        for name in self.names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qid.append(self.model.jnt_qposadr[jid])
            vid.append(self.model.jnt_dofadr[jid])
            aid.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
        self.qpos_ids = np.array(qid)
        self.qvel_ids = np.array(vid)
        self.act_ids = np.array(aid)

        arm = cfg.get("dof_armature")
        if arm is not None:
            self.model.dof_armature[self.qvel_ids] = np.array(arm, np.float64)
        frc = cfg.get("dof_friction")
        if frc is not None:
            self.model.dof_frictionloss[self.qvel_ids] = np.array(frc, np.float64)

        self.root_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.set_pose()
        if self.gantry > 0:
            z = [self.data.xpos[b, 2] for b in self._rope_bodies]
            self._rope_ref_z = sum(z) / len(z)
            print(f"  gantry safety rope: shoulder @ {self._rope_ref_z:.2f}m, {self._rope_ref_z - self._rope_slack:.2f}m active")

        # 窗口
        import cv2 as cv
        self.cv2 = cv
        self.win = "HoloWbt FSM Simulator"
        cv.namedWindow(self.win, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.win, 960, 600)

    # ---- 辅助 ----

    def set_pose(self, q=None, dq=None):
        import mujoco
        if q is None:
            q = self.default.copy()
        if dq is None:
            dq = np.zeros(J)
        self.data.qpos[:7] = [0, 0, 0.75, 1, 0, 0, 0]
        self.data.qpos[self.qpos_ids] = q
        self.data.qvel[:6] = 0
        self.data.qvel[self.qvel_ids] = dq
        mujoco.mj_forward(self.model, self.data)

    def set_pose_from_ref(self, frame):
        if frame < 0 or frame >= self.ref_data.shape[0]:
            frame = 0
        r = self.ref_data[frame]
        self.set_pose(q=r[:J].copy(), dq=r[J:2*J].copy() * 0.5)

    def snap(self) -> tuple:
        import mujoco
        q = self.data.qpos[self.qpos_ids].copy()
        dq = self.data.qvel[self.qvel_ids].copy()
        quat = self.data.xquat[self.root_body].copy()
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.root_body, vel6, 1)
        return q, dq, quat, vel6[:3]

    def compute_obs(self, q, dq, quat, ang, ref) -> np.ndarray:
        R_torso = forward_kinematics(self.chain, q)
        T = quat_wxyz_to_mat(quat) @ R_torso
        rq = ref[2*J + ANCHOR*7 + 3: 2*J + ANCHOR*7 + 7]
        al = yaw_from_quat(mat_to_quat(T)) @ yaw_from_quat(rq).T
        rel = T.T @ (al @ quat_wxyz_to_mat(rq))
        return np.concatenate([
            self.last_action, ang, q - self.default,
            dq, ref[:J], ref[J:2*J], rel[:, :2].reshape(-1),
        ]).astype(np.float32)

    # ---- 状态执行 ----

    def _tick_passive(self, _dt: float) -> None:
        import mujoco
        torque = -np.full(J, 3.0) * self.data.qvel[self.qvel_ids]
        self.data.ctrl[self.act_ids] = np.clip(torque, -self.limits, self.limits)
        mujoco.mj_step(self.model, self.data)

    def _tick_fixstand(self, _dt: float) -> None:
        import mujoco
        t = self.state_time
        a = min(t / 3.0, 1.0)
        target = (1 - a) * self.pose_start + a * self.pose_target
        kp = np.array(self.cfg.get("kp", self.kp), np.float64)
        kd = np.array(self.cfg.get("kd", self.kd), np.float64)
        torque = kp * (target - self.data.qpos[self.qpos_ids]) - kd * self.data.qvel[self.qvel_ids]
        self.data.ctrl[self.act_ids] = np.clip(torque, -self.limits, self.limits)
        mujoco.mj_step(self.model, self.data)
        # 锚定水平位置，避免漂移（首次进入时记录当前位置）
        if self._anchor_xy is None:
            self._anchor_xy = self.data.qpos[:2].copy()
        self.data.qpos[:2] = self._anchor_xy
        self.data.qvel[:2] = 0.0

    def _tick_prepose(self, _dt: float) -> None:
        import mujoco
        t = self.state_time
        a = min(t / 4.0, 1.0)
        target = (1 - a) * self.pose_start + a * self.pose_target
        ks = 3.0
        kp = self.kp * ks
        kd = self.kd * np.sqrt(ks)
        torque = kp * (target - self.data.qpos[self.qpos_ids]) - kd * self.data.qvel[self.qvel_ids]
        self.data.ctrl[self.act_ids] = np.clip(torque, -self.limits, self.limits)
        mujoco.mj_step(self.model, self.data)
        # 锚定水平位置，避免漂移
        if self._anchor_xy is not None:
            self.data.qpos[:2] = self._anchor_xy
            self.data.qvel[:2] = 0.0

    def _tick_holo(self, _dt: float) -> None:
        import mujoco
        fi = int(self.state_time * 50)
        if fi >= self.frame_count:
            self._trans(S_VELOCITY, f"end at frame {fi}")
            return
        if self.state_time > 2.0 and self.data.qpos[2] < 0.25:
            self.fell = True

        self._holo_tick += 1
        do_inf = self._holo_tick % 4 == 1
        q, dq, quat, ang = self.snap()
        ref = self.ref_data[min(fi, self.frame_count - 1)]

        if do_inf:
            if not self.policy_ok:
                R_t = forward_kinematics(self.chain, q)
                tq = mat_to_quat(quat_wxyz_to_mat(quat) @ R_t)
                rq = ref[2*J + ANCHOR*7 + 3: 2*J + ANCHOR*7 + 7]
                self.alignment = yaw_from_quat(tq) @ yaw_from_quat(rq).T
                self.policy_ok = True
                self.last_action.fill(0)
                self.last_target = q.copy()

            o = self.compute_obs(q, dq, quat, ang, ref)
            a32 = self.ort.run(["actions"], {"obs": o.reshape(1, O)})[0][0]
            a = a32.astype(np.float64)
            if not np.all(np.isfinite(a)):
                print("  WARN: non-finite action → damping")
                self._damping()
                return

            tgt = np.clip(self.default + self.scale * a,
                          self.lo + 0.01, self.hi - 0.01)
            tgt = np.clip(tgt, self.last_target - 0.16, self.last_target + 0.16)
            center = q + self.kd * dq / np.maximum(self.kp, 1e-10)
            radius = self.limits / np.maximum(self.kp, 1e-10)
            tgt = np.clip(tgt, center - radius, center + radius)
            tgt = np.clip(tgt, self.lo + 0.01, self.hi - 0.01)
            self.clipped += int(np.sum(np.abs(tgt - (self.default + self.scale * a)) > 1e-5))
            self.last_action = a.copy()
            self.last_target = tgt.copy()
        else:
            tgt = self.last_target

        torque = self.kp * (tgt - q) - self.kd * dq
        self.data.ctrl[self.act_ids] = np.clip(torque, -self.limits, self.limits)
        mujoco.mj_step(self.model, self.data)

    def _tick_velocity(self, _dt: float) -> None:
        import mujoco
        kp = self.kp * 0.25
        q = self.data.qpos[self.qpos_ids]
        torque = kp * (self.pose_target - q)
        self.data.ctrl[self.act_ids] = np.clip(torque, -self.limits, self.limits)
        mujoco.mj_step(self.model, self.data)

    def _damping(self) -> None:
        import mujoco
        torque = -np.full(J, 3.0) * self.data.qvel[self.qvel_ids]
        self.data.ctrl[self.act_ids] = np.clip(torque, -self.limits, self.limits)
        mujoco.mj_step(self.model, self.data)

    def _trans(self, new: int, reason: str = "") -> None:
        if new == self.state:
            return
        old = STATE_NAMES.get(self.state, f"s{self.state}")
        n = STATE_NAMES.get(new, f"s{new}")
        print(f"  [{time.strftime('%H:%M:%S')}] {old} → {n}  {reason}")
        # 进入需要插值的状态时，将 pose_start 更新为当前姿态，避免跳变
        if new == S_FIXSTAND or new == S_HOLO_PREPOSE:
            self.pose_start = self.data.qpos[self.qpos_ids].copy()
            self._anchor_xy = self.data.qpos[:2].copy()
        self.state = new
        self.state_time = 0.0
        self.policy_ok = False
        self._holo_tick = 0

    # ---- 键盘 / 自动 输入 ----

    def _get_input(self) -> tuple:
        """返回 (L2, R1, UP, RIGHT, X, A, B, ESC)"""
        if self.auto:
            return self._auto_input()
        try:
            import ctypes
            u = ctypes.windll.user32
            lshift = (u.GetKeyState(0xA0) >> 15) & 1
            rshift = (u.GetKeyState(0xA1) >> 15) & 1
            rctrl = (u.GetKeyState(0xA3) >> 15) & 1
            up = (u.GetKeyState(0x26) >> 15) & 1
            right = (u.GetKeyState(0x27) >> 15) & 1
            xk = (u.GetKeyState(ord("X")) >> 15) & 1
            ak = (u.GetKeyState(ord("A")) >> 15) & 1
            bk = (u.GetKeyState(ord("B")) >> 15) & 1
            esc = (u.GetKeyState(0x1B) >> 15) & 1
            return (lshift or rshift), rctrl, up, right, xk, ak, bk, esc
        except Exception:
            return False, False, False, False, False, False, False, False

    def _auto_input(self) -> tuple:
        """自动演示：按日程发送按键事件"""
        t = self.elapsed
        L2, R1, UP, RIGHT, X, A, B = False, False, False, False, False, False, False

        # 调度表 (brief pulses, not held down):
        # t=0      起始 PASSIVE
        # t=1.0    LShift+Up       (momentary) → FixStand
        # t=4.0    LShift hold (准备 2s 等待)
        # t=6.0    LShift(2s)+Right(momentary) → Prepose
        # t=9.5    RCtrl+X         (momentary) → Box 阶段
        # t=12.0   (Box 自动完成    → Velocity)
        # t=14.5   (Box 自动完成 → Velocity)
        # t=15.5   LShift+B        (momentary) → Passive
        # t=17.0   结束

        if 1.0 <= t < 1.005:
            L2, UP = True, True
        elif 4.0 <= t < 6.005:
            L2 = True               # hold for 2s → trigger at t=6.0
            if 6.0 <= t < 6.005:
                RIGHT = True       # brief pulse: L2(2s)+Right
        elif 9.5 <= t < 9.505:
            R1, X = True, True
        elif 15.5 <= t < 15.505:
            L2, B = True, True

        return L2, R1, UP, RIGHT, X, A, B, False

    # ---- 主循环 ----

    def run(self, max_s: float = 60.0) -> None:
        import mujoco
        import cv2 as cv

        r0 = self.ref_data[0]
        self.pose_start = self.data.qpos[self.qpos_ids].copy()
        self.pose_target = r0[:J].copy().astype(np.float64)

        renderer = mujoco.Renderer(self.model, height=600, width=960)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 3.0, 135, -18
        step = 0

        _tick = {
            S_PASSIVE: self._tick_passive,
            S_FIXSTAND: self._tick_fixstand,
            S_HOLO_PREPOSE: self._tick_prepose,
            S_HOLO_BOX: self._tick_holo,
            S_VELOCITY: self._tick_velocity,
        }

        # 初始化录制器
        if self.record_path:
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            self._writer = cv.VideoWriter(self.record_path, fourcc, self._fps, (960, 600))
            print(f"  recording to {self.record_path}")

        print("FSM ready:")
        print("  LShift+Up       → FixStand")
        print("  LShift(2s)+R    → Prepose")
        print("  RCtrl+X         → Run / Early End")
        print("  RCtrl+A         → Passive")
        print("  LShift+B/Q/Esc  → Quit")
        if self.auto:
            print("  [AUTO DEMO MODE]")
        print()

        while self.running and self.elapsed < max_s:
            dt = self.model.opt.timestep

            L2, R1, UP, RIGHT, X, A, B, ESC = self._get_input()

            if ESC:
                self.running = False

            if self.state == S_PASSIVE:
                if L2 and UP:
                    self._trans(S_FIXSTAND, "LShift+Up")

            elif self.state == S_FIXSTAND:
                if L2 and B:
                    self._trans(S_PASSIVE, "LShift+B")
                elif L2 and RIGHT:
                    if self._l2_hold_start >= 0 and (time.time() - self._l2_hold_start) > 2.0:
                        self._trans(S_HOLO_PREPOSE, "LShift(2s)+Right")
                else:
                    if L2 and self._l2_hold_start < 0:
                        self._l2_hold_start = time.time()
                    elif not L2:
                        self._l2_hold_start = -1

            elif self.state == S_HOLO_PREPOSE:
                if R1 and A:
                    self._trans(S_PASSIVE, "RCtrl+A")
                elif R1 and X:
                    self._trans(S_HOLO_BOX, "RCtrl+X")

            elif self.state == S_HOLO_BOX:
                if R1 and A:
                    self._trans(S_PASSIVE, "RCtrl+A")
                elif R1 and X:
                    self._trans(S_VELOCITY, "RCtrl+X early")
                elif self.fell:
                    self._trans(S_PASSIVE, "fall detected")

            elif self.state == S_VELOCITY:
                if L2 and B:
                    self._trans(S_PASSIVE, "LShift+B")

            _tick[self.state](dt)

            # 吊架: 在骨盆施加向上力补偿部分重力
            # 吊架: 安全绳 骨盆低于阈值时施加弹力
            if self.gantry > 0:
                h = sum(self.data.xpos[b, 2] for b in self._rope_bodies) / len(self._rope_bodies)
                dz = max(0.0, (self._rope_ref_z - self._rope_slack) - h)
                vz = sum(self.data.cvel[b, 5] for b in self._rope_bodies) / len(self._rope_bodies)
                fz = self._rope_k * dz - self._rope_d * vz
                f = max(0.0, fz) * self.gantry / len(self._rope_bodies)
                for b in self._rope_bodies:
                    self.data.xfrc_applied[b, 2] = f

            self.elapsed += dt
            self.state_time += dt
            step += 1

            if step % 2 == 0:
                cam.lookat[:] = self.data.xpos[self.root_body]
                cam.lookat[2] = 0.6
                renderer.update_scene(self.data, camera=cam)
                frame = cv.cvtColor(renderer.render(), cv.COLOR_RGB2BGR)

                tilt_deg = np.degrees(np.arccos(np.clip(
                    self.data.xmat[self.root_body].reshape(3, 3)[2, 2], -1, 1)))
                color = STATE_COLORS.get(self.state, (200, 200, 200))
                cv.rectangle(frame, (0, 0), (960, 65), (20, 20, 20), -1)

                lines = [
                    f"State: {STATE_NAMES.get(self.state, '?')}  ({self.state_time:.1f}s)",
                    f"H: {self.data.qpos[2]:.3f}m | Tilt: {tilt_deg:.1f}° | Frame: {int(self.state_time*50):3d}/{self.frame_count}",
                    f"Clipped: {self.clipped}",
                ]
                for i, txt in enumerate(lines):
                    cv.putText(frame, txt, (15, 22 + i * 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                if self.auto:
                    prog = min(self.elapsed / 15.0, 1.0)
                    bar_w = int(prog * 940)
                    cv.rectangle(frame, (10, 580), (950, 595), (40, 40, 40), -1)
                    cv.rectangle(frame, (10, 580), (10 + bar_w, 595), (60, 200, 60), -1)
                    cv.putText(frame, f"{prog*100:.0f}%", (460, 593),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                hints = {
                    S_PASSIVE: "LShift+Up → FixStand",
                    S_FIXSTAND: "LShift(2s)+Right→Prepose | LShift+B→Passive",
                    S_HOLO_PREPOSE: "RCtrl+X→Run | RCtrl+A→Passive",
                    S_HOLO_BOX: "Auto→Vel | RCtrl+X→Early",
                    S_VELOCITY: "LShift+B→Passive",
                }
                cv.putText(frame, hints.get(self.state, ""), (15, 560),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 220), 1)
                cv.imshow(self.win, frame)

                if self._writer:
                    self._writer.write(frame)

                key = cv.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self.running = False

            # 自动模式：16 秒后结束
            if self.auto and self.elapsed >= 15.0:
                self.running = False

        renderer.close()
        cv.destroyAllWindows()
        if self._writer:
            self._writer.release()
            print(f"  video saved: {self.record_path}")
        print(f"Simulation ended after {self.elapsed:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description="MuJoCo HoloWbt FSM Simulator")
    p.add_argument("--package", "-p",
                   default="deploy/robots/g1_29dof/config/policy/holo_wbt/czy_box")
    p.add_argument("--urdf")
    p.add_argument("--max-s", type=float, default=60.0)
    p.add_argument("--record", help="output video path (.mp4)")
    p.add_argument("--auto", action="store_true", help="auto-demo mode")
    p.add_argument("--gantry", type=float, default=0.0,
                   help="safety gantry compensation (0-1.0, e.g. 0.8 = 80% weight offload)")
    a = p.parse_args()
    HoloWbtFSMSim(a.package, a.urdf, record=a.record, auto=a.auto, gantry=a.gantry).run(a.max_s)


if __name__ == "__main__":
    main()

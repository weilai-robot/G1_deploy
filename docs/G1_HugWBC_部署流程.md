# G1 EDU U2 29DOF：HugWBC（R41/R42）模型部署流程

编写日期：2026-11-02。分支 `yanni-G1-walk-hugWBC`。实现依据 [HugWBC_G1_29DOF_部署适配方案.md](../HugWBC_G1_29DOF_部署适配方案.md)（该文档由规划 AI 编写并核对了训练源码与模型张量；本文档记录执行 AI 的实际实现与实测结果）。

## 当前状态与使用范围

| 阶段 | 状态 |
|---|---|
| ONNX 导出（R41/R42） | ✅ 已执行，PyTorch↔ONNX 实测误差见 §5 |
| 部署包校验（哈希/契约/PD/映射） | ✅ 两个包均通过 `tools/check_hugwbc_package.py` |
| Python 参考核心数值回放 | ✅ 已执行（含真实 rollout 相位锚点，见 §6） |
| C++ 数值对照（hugwbc_offline） | ⚠️ 代码已完成；本机（Windows）无 C++ 工具链，未执行——须在 Linux 构建机上运行 `tools/verify_hugwbc.py --offline-binary` |
| C++ 闭环仿真 | ❌ 未执行 |
| 真机验证 | ❌ 未执行（须人批准、人操作） |
| 启停/倒退/侧移/转向 | ❌ 未实现、未验证；当前只有固定前行测试 profile |

**不要在无闭环仿真与真机批准的情况下运行。** 吊装防跌落保护必须全程保持。

## 1. 模型身份与选择

| 版本 | 来源 checkpoint（SHA256） | 迭代 | 训练源提交 | 选择方式 |
|---|---|---:|---|---|
| R41 | `2e468ff1…f6b51`（model_12600_g1r41.pt） | 12600 | `b219b94b…`（hugwbc-g1 分支） | `--config config/hugwbc_r41.yaml` |
| R42 | `49333714…7cb04`（model_13200_g1r42.pt） | 13200 | `4c1234ba…`（hugwbc-g1 分支） | `--config config/hugwbc_r42.yaml` |

- 两版共用同一控制器语义：腿腰 15 维用策略输出，8 个臂关节用步态相位参考完全覆盖，interrupt 命令位恒为 0。
- 版本选择**只**由 `--config` 指定的 YAML 中的 `policy_dir` 决定；代码不会按文件名/时间挑"最新"模型。
- 两包目录独立：`config/policy/hugwbc/r41/` 与 `r42/`，各自含 `policy.onnx`、`deploy.yaml`、`manifest.json`，互不共享。

## 2. 代码结构（本次新增/修改）

```text
tools/export_hugwbc.py                     新增：无 Isaac Gym 导出（重建 mem_encoder/state_estimator/low_level_net）
tools/check_hugwbc_package.py              新增：依赖极简的包校验（哈希+契约字段）
tools/verify_hugwbc.py                     新增：PyTorch↔ONNX↔Python参考核心↔C++ 分层比对
tools/select_package_checks.py             新增：按 FSM 配置选择包检查器
tools/build_g1_deploy.sh                   修改：--config 选择配置；检查器随配置走（不再无条件跑 BeyondMimic 检查）
tools/build_g1_hugwbc.sh                   新增：HugWBC 构建包装（r41|r42，默认 r41）
deploy/include/hugwbc/core.h               新增：纯 C++ 数值核心（观测/时钟/参考臂/合成/PD守护）
deploy/include/hugwbc/package.h            新增：deploy.yaml 加载+严格校验+SHA256 身份
deploy/include/hugwbc/runner.h             新增：ORT 推理（[1,5,88]→[1,23] 严格检查）
deploy/robots/g1_29dof/include/State_HugWBC.h   新增：独立 FSM 状态
deploy/robots/g1_29dof/src/State_HugWBC.cpp     新增
deploy/robots/g1_29dof/config/hugwbc_r41.yaml   新增：专用配置（不覆盖 config.yaml）
deploy/robots/g1_29dof/config/hugwbc_r42.yaml   新增
deploy/robots/g1_29dof/config/policy/hugwbc/r41/  新增：导出产物
deploy/robots/g1_29dof/config/policy/hugwbc/r42/  新增：导出产物
deploy/robots/g1_29dof/main.cpp            修改：注册状态；HugWBC 也要求显式 --network
deploy/robots/g1_29dof/CMakeLists.txt      修改：新增 hugwbc_runtime / hugwbc_offline 目标
deploy/tools/hugwbc_offline.cpp            新增：无 DDS 数值回放（JSON-lines stdin/stdout）
docs/G1_HugWBC_部署流程.md                  新增：本文
```

既有 Velocity、Mimic、BeyondMimic 状态与 `config.yaml` 未改动；原有构建目标保留。

## 3. 控制契约（导出包 deploy.yaml 的字段来源）

### 3.1 关节映射（23 模型维 → 29 SDK 电机）

```text
model_to_sdk = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,22,23,24,25]
腕部保持     = SDK 19/20/21（左 roll/pitch/yaw）+ 26/27/28（右），目标 0 rad
```

六个腕关节在训练资产（g1_29dof_anneal_23dof.urdf）中为 `type="fixed"`，故腕部不属于模型契约；部署用位置保持（kp=40/kd=10，取自本分支 FixStand 腕行——**这是部署保持参数，不是训练值**）。实施时须核对真机编码器零位与训练资产零位一致性并记录实际腕角。

### 3.2 每帧 88 维观测（先缩放，后整帧 clip ±100）

| 切片 | 内容 | 缩放 |
|---|---|---|
| [0:3] | pelvis 角速度（体系） | ×0.25 |
| [3:6] | pelvis 系重力投影 `R_WP^T·[0,0,-1]` | ×1 |
| [6:29] | `q23 − q0` | ×1 |
| [29:52] | `dq23` | ×0.05 |
| [52:75] | 上一 tick 混合动作（替换臂后、clip 前） | ×1 |
| [75:86] | cmd11 × [2,2,0.25,1,1,1,0.15,2,0.5,0.5,1] | — |
| [86:88] | `sin(2π·remap(相位))` 左/右 | — |

### 3.3 步态时钟（50 Hz，已对 16000 行真实 rollout 验证）

```text
g ← remainder(g + 0.02·cmd3, 1)          # 每策略 tick 推进一次
phase_left  = remainder(g + cmd4, 1)      # phase_right = g
clock       = sin(2π·remap(phase, cmd5))  # remap: u<d→u·0.5/d；u>d→0.5+(u−d)·0.5/(1−d)；u=d→原值
```

站立时左右相位/时钟置 0（全局 g 照常推进）；摆臂参考使用**未重映射**相位，左肩用右脚相位、右肩用左脚相位，与策略同一份相位。

### 3.4 双臂参考与动作合成

```text
A=0.09178921571001411  δ=0.30418359554562746  speed_scale=clamp(cmd_vx/0.5,−1,1)  walking=!standing
ref_L_pitch  = 0.3 − A·cos(2π·phase_right − δ)·speed_scale·walking
ref_R_pitch  = 0.3 − A·cos(2π·phase_left  − δ)·speed_scale·walking
ref_roll = ±0.27（左+右−）  ref_yaw = 0  ref_elbow = 0.97
hybrid[arm] = (ref − q0_arm)/0.25；q_target = q0 + 0.25·clip(hybrid, ±100)
历史存 hybrid（clip 前）；无参考速度前馈（原执行 feedforward=false）
```

### 3.5 PD 与力矩（训练/评估值，非硬件额定）

| 关节组 | Kp | Kd | τ 上限 (Nm) |
|---|---:|---:|---:|
| hip pitch/roll/yaw | 100 | 2.5 | 88 |
| knee | 150 | 3 | 140 |
| ankle pitch/roll | 20 | 1 | 45 |
| waist yaw/roll/pitch | 150 | 3 | 88 |
| shoulder pitch | 90 | 2 | 25 |
| shoulder roll | 60 | 1 | 25 |
| shoulder yaw | 20 | 0.4 | 25 |
| elbow | 60 | 1 | 25 |

部署侧守护（显式包装，可关）：把目标投影到 `q + (Kd/Kp)·dq ± τ/Kp` 内并与位置限位（URDF 限位 ±0.01）取交集；腕部无训练值，不做力矩守护。`max_target_speed` 默认 0（关闭，训练无此限）。

### 3.6 固定前行测试命令

`cmd11 = [0.5, 0, 0, 1.5, 0.5, 0.5, 0.1, 0, 0, 0, 0]`（vx, vy, wz, 步频, 相位偏移, 占空比, 抬足高, 身高, pitch, 腰, interrupt=0）。与接受评估协议一致；其余命令模式未开放。

## 4. 导出与构建

导出（Windows/Linux 均可，需 torch+onnx+onnxruntime+pyyaml，无需 Isaac Gym）：

```bash
python tools/export_hugwbc.py --checkpoint <model_12600_g1r41.pt> --task g1r41 \
  --output deploy/robots/g1_29dof/config/policy/hugwbc/r41
python tools/export_hugwbc.py --checkpoint <model_13200_g1r42.pt> --task g1r42 \
  --output deploy/robots/g1_29dof/config/policy/hugwbc/r42
```

构建（Linux 部署机）：

```bash
tools/build_g1_hugwbc.sh r41     # 或 r42；内部调用 build_g1_deploy.sh --config config/hugwbc_r41.yaml
```

运行（真机前必须先过 §5/§6 全部检查）：

```bash
./build/g1_ctrl --config config/hugwbc_r41.yaml --network <iface> [--domain N]
# Passive → [L2+Up] FixStand（3 秒插值到 HugWBC 准备姿态）→ [L2 按住 2s + Right] 进入 HugWBC
# [L2+B] 或 Ctrl+C → 阻尼；全程保持吊装防跌落保护
```

## 5. 实测导出结果（2026-11-02，本机）

工具链：Python 3.10.0rc2 / torch 2.6.0+cpu / onnx 1.21.0 / onnxruntime 1.23.2，opset 17，输入 `obs_history [1,5,88]`、输出 `actions [1,23]`。

| 包 | checkpoint SHA（前 16） | policy.onnx SHA（前 16） | PyTorch↔ONNX 最大绝对误差（真实量级输入） | （±100 极端输入） |
|---|---|---|---:|---:|
| r41 | `2e468ff18b5d3479` | `836b2f96ea558385` | **3.81e-06** | 3.66e-04 |
| r42 | `4933371459e0e26e` | `29f777b00cad453c` | **4.77e-06** | 4.17e-04 |

- parity 门限沿用本仓库导出器惯例（rtol/atol 2e-4），实测误差远低于门限；极端输入误差大属 float32 放大，正常量级输入误差为部署容差依据。
- `tools/check_hugwbc_package.py`：两包哈希、契约字段（维度/映射/PD/臂参数/缩放/裁剪）全部通过。

## 6. 数值验证结果与未验证项

`tools/verify_hugwbc.py`（R41/R42 均已跑）：

| 检查 | 结果 |
|---|---|
| checkpoint SHA == 冻结值 | ✅ 两版一致 |
| 重建 actor ↔ ONNX（64 样本） | ✅ 最大误差 1.14e-05（r41）/ 9.54e-06（r42） |
| 真实 rollout 锚点（16000 行，exp-20261030-004 baseline-hybrid seed1） | ✅ 相位推进 = 0.02·cmd3（误差 ≤3.1e-08），左右相位差 = 半周期（误差 ≤3.0e-08） |
| Python 参考核心：站立时钟置零/行走回绕 100 tick | ✅ |
| C++ 对照（`--offline-binary`） | ⚠️ 未执行：本机无 C++ 工具链 |
| C++ 闭环仿真 | ❌ 未执行 |
| 真机（pelvis IMU 源、腕部零位标定） | ❌ 未执行 |

**C++ 对照在 Linux 构建机上补跑的命令**：

```bash
cmake -S deploy/robots/g1_29dof -B deploy/robots/g1_29dof/build -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=deploy/thirdparty/onnxruntime-linux-<arch>-1.22.0 -DBUILD_ROBOT_CONTROLLER=OFF
cmake --build deploy/robots/g1_29dof/build --target hugwbc_offline
python tools/verify_hugwbc.py --offline-binary deploy/robots/g1_29dof/build/hugwbc_offline --report tools/hugwbc_verify_report.json
```

## 7. 硬件注意事项（真机前必读）

1. **IMU 来源**：actor 输入是 **pelvis** 姿态/角速度。当前实现读 `imu_state()`（默认 `imu_source: pelvis`）。上真机前必须核对所用 SDK/仿真桥该字段确实属于 pelvis；若只有 torso IMU，deploy.yaml 改 `imu_source: torso`（实现里已含腰链 yaw→roll→pitch 的 pelvis 还原），并按 §6 用"仅动腰"记录验证 pelvis 观测不随躯干误转。四元数按接入层 wxyz 读取（Isaac Gym 内部 xyzw 的差异已在核心内处理约定）。
2. **腕部**：目标 0 rad 位置保持；上电核对编码器零位与训练资产一致性；日志记录实际腕角。
3. **踝/腰 PR 语义**：训练为 pitch/roll 语义，接入前核对 SDK 串联 PR 模式，不得当并联 A/B 角。
4. **起步**：进入 HugWBC 时动作历史清零、步态相位从 0 起步、5 帧历史用首帧真实观测填充——这是与训练 reset 不同的显式部署包装，闭环仿真未验证前不得上机。`max_start_error: 0.6`（rad）为未实测占位阈值。
5. **启停/转向**：未实现。不要把 Velocity 手柄命令范围开放给 HugWBC。
6. **退出**：异常→阻尼（kd=3）→Passive；阻尼不是已训练的制动能力。

## 8. 日志

State_HugWBC 每 50 策略 tick（约 1 Hz）在策略线程输出：tick、推理耗时 ms、g/相位/时钟、cmd_vx、standing、raw/hybrid 样本、臂参考 pitch、守护触发数、SDK 目标样本；构造时打印包名、checkpoint SHA、policy SHA、IMU 源与命令；退出原因必记。完整逐 tick 数值走离线回放路径（hugwbc_offline + verify_hugwbc）。

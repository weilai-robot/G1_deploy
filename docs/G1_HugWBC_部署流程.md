# G1 EDU U2 29DOF：HugWBC（R41/R42）模型部署流程

编写日期：2026-11-02（rev3，复审 7956a48 并修正剩余问题）。分支 `yanni-G1-walk-hugWBC`。实现依据 [HugWBC_G1_29DOF_部署适配方案.md](../HugWBC_G1_29DOF_部署适配方案.md)。Python 结果见 `tools/hugwbc_verify_report.json`，实际 Linux C++ 构建及跨平台回放结果见 `tools/hugwbc_cpp_review_report.json`。

## 当前状态与使用范围

| 阶段 | 状态 |
|---|---|
| ONNX 导出（R41/R42） | ✅ 已执行，PyTorch↔ONNX 实测误差见 §5 |
| 部署包校验（哈希/契约/PD/映射） | ✅ 两个包均通过 `tools/check_hugwbc_package.py`（含显式包路径调用方式） |
| Python 参考核心数值回放 | ✅ 已执行（含真实 rollout 相位锚点，见 §6） |
| 审核回归（F1/F4/F5/F6/F7） | ✅ `verify_hugwbc.py regressions` 全部通过（§6.1） |
| C++ 离线编译与数值对照 | ✅ 已在 tare-codex 隔离目录实际编译；两包逐拍对照、核心回归和故障拒绝通过，见 §6.2 |
| 完整 g1_ctrl 构建 | ⚠️ 未完成；审核构建机的配置检查停在缺少 spdlog，机器人主程序未运行 |
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
bash tools/build_g1_hugwbc.sh r41     # 或 r42；包装脚本通过 bash 调用内部构建脚本，不依赖可执行位
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
| 审核回归 F1/F4/F5/F6/F7 | ✅ 全部通过（§6.1） |
| C++ 跨平台回放 | ✅ 实际 Linux 二进制与 Windows Python/ONNX 对照，见 §6.2；不同于在 Linux 运行完整 Python 验证器 |
| C++ 闭环仿真 | ❌ 未执行 |
| 真机（pelvis IMU 源、腕部零位标定） | ❌ 未执行 |

### 6.1 审核回归（rev2 新增，verify_hugwbc.py `regressions` 字段）

| 回归 | 内容 | 结果（两版相同） |
|---|---|---|
| F1 历史顺序 | 连续 8 帧、t≥5 后检查 5 帧槽位 = [t-4, t-3, t-2, t-1, t]（淘汰最旧帧） | ✅ ok |
| F4 hybrid 回填 | 第 t+1 拍观测 [52:75] == 第 t 拍 clip 前 hybrid（含参考臂槽位） | ✅ ok |
| F5 空交集拒绝 | 位置/速度/力矩约束无交集时抛出异常，不返回或下发该拍目标；FSM 进入阻尼/Passive 路径 | ✅ Python 与 C++ 均拒绝 |
| F6 torso IMU 补偿 | 正确方向旋转 torso gyro，再减腰相对角速度；额外覆盖非零 yaw/roll/pitch 和倾斜、运动的 pelvis | ✅ Python 与 C++ 三轴用例通过 |
| F7 腕目标索引 | 六腕写入非零保持目标 [0.1,-0.1,0.05,-0.05,0.02,-0.02]，逐 SDK 槽位读回 | ✅ ok |
| 速度限制初始化 | 第一拍无上一目标时不施加速度界，后续才使用上一目标 | ✅ Python 与 C++ 一致 |

**Linux 复验命令**（路径按实际安装填写；离线检查不启动机器人）：

```bash
cmake -S deploy/robots/g1_29dof -B deploy/robots/g1_29dof/build -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="$PWD/deploy/thirdparty/onnxruntime-linux-x64-1.22.0" -DBUILD_ROBOT_CONTROLLER=OFF
cmake --build deploy/robots/g1_29dof/build --target hugwbc_offline hugwbc_core_checks
for variant in r41 r42; do
  deploy/robots/g1_29dof/build/hugwbc_core_checks "deploy/robots/g1_29dof/config/policy/hugwbc/$variant"
done
python tools/verify_hugwbc.py --training-root /path/to/hugwbc-g1 \
  --offline-binary deploy/robots/g1_29dof/build/hugwbc_offline --report tools/hugwbc_verify_report.json
```

### 6.2 rev3 实际 C++ 验证

构建机：tare-codex，独立目录 `/tmp/hugwbc-deploy-review-7956a48/`；关闭 `BUILD_ROBOT_CONTROLLER` 和 `BUILD_BEYONDMIMIC_OFFLINE`，使用包内 ONNX Runtime 与临时目录安装的 yaml-cpp。没有全局安装依赖或运行 DDS。

证据：`tools/hugwbc_cpp_review_report.json` 的 `core_checks`、`negative_tests`、`parity`；完整输入和输出保存在仓库外 `.review-artifacts/hugwbc-deploy-7956a48/`。

- 每个模型：行走回放 100 拍、站立/行走切换及显式重置回放 60 拍，逐拍比较观测、历史、动作、参考臂和 SDK 目标。数字来源：报告各 `parity.*.ticks` 字段。
- 每个模型：C++ 核心回归通过；无效 IMU、约束空交集、错误包格式、损坏模型均拒绝，且无目标输出。
- 感知/命令/时钟槽位保持严格独立比较；上一动作槽位沿用已有 ONNX 浮点绝对容差，避免跨平台推理差异经历史回填后导致误判。具体容差见 `verify_hugwbc.py:CPP_ATOL`，实测误差见报告，不能据此推导真机精度。
- 此回放不是闭环仿真，也不证明完整控制器已编译或真机步态已经通过。

## 7. rev1 审核结论与逐项修复（a075fda 审核，rev2 修复）

外部审核（`G1_HugWBC_部署执行审核_a075fda.md`，证据目录 `.review-artifacts/hugwbc-deploy-a075fda/`）判定 rev1 不通过并给出 F1-F7；逐项处理：

| 项 | 审核判定 | 修复 | 验证 |
|---|---|---|---|
| F1 P1 历史移位方向错（`copy_backward` 保旧丢新） | 成立 | `core.h` 改为训练 `ObservationBuffer.insert` 语义：`copy(begin+OBS, end, begin)` 淘汰最旧帧 | 回归 F1（§6.1） |
| F2 P1 SHA 类编译错（原生数组花括号赋值）+ core.h 缺 `<cstdint>` | 成立 | `h_` 改 `std::array<std::uint32_t,8>`；core.h 自含 `<cstdint>` | 实际离线二进制编译与模型哈希校验通过 |
| F3 P1 构建入口把"脚本+参数"整行当文件名；校验器按目录名推断任务失败 | 成立 | 选择器改 TAB 分隔；`build_g1_deploy.sh` 按 TAB 拆分并透传包路径（防 CRLF）；`check_hugwbc_package.py` 接受 r41/r42 与 g1r41/g1r42 两种拼写 | 显式包路径调用两包 PASS |
| F4 P2 Python 参考核心未回填上一拍 hybrid | 成立 | `complete()` 末尾回填 `last_hybrid`（并补 `last_target` 状态镜像速度限幅）；`verify` 断言观测动作切片 == 上一拍 hybrid | 回归 F4 |
| F5 P2 限幅顺序使力矩守护可被位置限幅覆盖且不报告 | 成立 | rev3 修正 rev2 的回落位置界行为：交集为空时拒绝整拍，异常进入已有故障退出路径；不假定固件采用训练限矩值 | Python、C++ 回归与离线故障拒绝 |
| F6 P2 torso IMU 只旋转陀螺、未减腰相对角速度 | 成立 | rev3 修正 rev2 陀螺旋转方向；先用 pelvis←torso 矩阵旋转，再扣除腰相对角速度；倾角门与 actor 同帧 | yaw 及三轴回归 |
| F7 P2 腕目标写入按 SDK 索引、读取按枚举序号 | 成立 | `complete()` 统一 `wrist_target[wrist_sdk_indices[w]]`（加载端同为 SDK 索引） | 回归 F7 |

逐拍 CSV：`log/hugwbc_ticks_<epoch_microseconds>.csv`，配置 `tick_log` 默认开；记录主机快照开始、推理结束、目标就绪单调时钟整数微秒，保留浮点数精度。**这些时间不是设备采样或 DDS 实际发送时间**。另记录原始输入姿态/陀螺、q/dq、pelvis 观测、完整动作与目标、相位、参考臂和限幅情况。写入发生在策略线程，文件 I/O 延迟仍可能影响策略线程；完整发送时序需要接入 DDS 发送层另行验证。

## 8. 硬件注意事项（真机前必读）

1. **IMU 来源**：actor 输入是 **pelvis** 姿态/角速度。当前实现读 `imu_state()`（默认 `imu_source: pelvis`）。上真机前必须核对所用 SDK/仿真桥该字段确实属于 pelvis；若只有 torso IMU，deploy.yaml 改 `imu_source: torso`（实现里已含腰链 yaw→roll→pitch 的 pelvis 还原 + 腰相对角速度扣除），并按 §6.1 用"仅动腰"记录验证 pelvis 观测不随躯干误转。倾角门与 actor 用同一 pelvis 帧计算。四元数按接入层 wxyz 读取（Isaac Gym 内部 xyzw 的差异已在核心内处理约定）。
2. **腕部**：目标 0 rad 位置保持；上电核对编码器零位与训练资产一致性；逐 tick 日志记录实际腕角。
3. **踝/腰 PR 语义**：训练为 pitch/roll 语义，接入前核对 SDK 串联 PR 模式，不得当并联 A/B 角。
4. **起步**：进入 HugWBC 时动作历史清零、步态相位从 0 起步、5 帧历史用首帧真实观测填充——这是与训练 reset 不同的显式部署包装，闭环仿真未验证前不得上机。`max_start_error: 0.6`（rad）为未实测占位阈值。
5. **启停/转向**：未实现。不要把 Velocity 手柄命令范围开放给 HugWBC。
6. **退出**：异常→阻尼（kd=3）→Passive；阻尼不是已训练的制动能力。
7. **保护约束无交集**：整拍拒绝，不更新发布目标，记录 `Empty target constraint intersection at SDK motor ...` 原因并进入异常退出路径；返回的正常 tick 中 `guard_violations` 恒为零。不能把固件限矩当作已验证的训练力矩限幅。

## 9. 日志

- **逐 tick CSV**（`log/hugwbc_ticks_<epoch>.csv`，`tick_log: true` 默认开，策略线程写入）：字段全列见 §7 末段；sim2real 分析以该文件为准。
- **1 Hz 控制台**（策略线程，每 50 tick）：tick、推理耗时 ms、g/相位/时钟、cmd_vx、standing、raw/hybrid 样本、臂参考 pitch、limited/violated、SDK 目标样本。
- **生命周期**：构造打印包名、checkpoint SHA、policy SHA、IMU 源与命令；约束无交集等退出原因必记。

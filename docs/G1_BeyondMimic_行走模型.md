# BeyondMimic G1-29DOF 行走模型

## 模型与配置

模型包在 `deploy/robots/g1_29dof/config/policy/beyondmimic/`，只适配冻结权重，不改变归一化、PD 参数或基线选择。

| 模型包 | checkpoint | FSM 配置 |
|---|---|---|
| `walk_baseline` | exp003 / model_3998.pt | `beyondmimic_walk.yaml` |
| `walk_contact` | exp004 / model_5997.pt | `beyondmimic_walk_contact.yaml` |

每包包含 `policy.onnx`、`gait.bin`、`deploy.yaml`、`manifest.json`。来源 commit、checkpoint/ONNX/参考 SHA 见 manifest。PT、视频及训练档案仍在 loop 的双模型归档，部署机不依赖该归档。

## 手柄与状态机

新增 `State_BeyondMimicWalk` 与旧的有限舞蹈状态独立。FixStand 使用本模型初始支撑帧关节角和 SDK 顺序 PD，不能换用旧舞蹈默认站姿。

| 操作 | 行为 |
|---|---|
| L2 + 上 | FixStand，过渡到当前模型初始支撑姿态 |
| 完成 FixStand、摇杆回中后 R1 + X | 进入持续的 BeyondMimicWalk |
| 左杆前推 | 训练范围内的前向速度；推杆越大速度越大 |
| 前推时拨右杆 | 行走中转向；右杆向右对应右转 |
| 左杆回中 | 在可达支撑相制动并持续保持策略站立 |
| 再次前推 | 从当前支撑相起步，策略历史与里程计连续 |
| L2 + B 或 Ctrl+C | 退出策略，发布阻尼 |

左杆左右和后推不请求侧移或倒退；不前推时右杆不请求原地转向。回中保持策略站立和 Passive 阻尼是不同操作。策略没有固定播放结束时间；指令切换和相位回绕不重置参考原点。换模型需退出控制器后选另一份配置，不支持热切换。

速度、转向和相位范围直接来自包 `deploy.yaml` 的 `cyclic_reference` 与 `training_command_curriculum`。前推映射采用训练的 `speed_scale_range`，转向采用 `command_limits[2]`。`command_deadzone: 0.1` 是显式输入配置选择，不是实测标定。

## 验证状态与部署阻塞

实际结果见 [beyondmimic_walk_validation.json](beyondmimic_walk_validation.json) 的 `completed_seconds`、`termination_reason`、误差及校验字段。状态 CSV 的压缩副本和冻结协议位于 `validation/walk/`。

已核对模型 SHA、内置归一化的 ONNX/Torch 一致性、C++/原生 Python 循环参考一致性、C++/MuJoCo FK 观测一致性，以及错误 IMU、非法指令、关节映射和损坏步态的拒绝行为。

**现有腿式里程计在两模型闭环中均累计漂移，触发保留的 anchor-error 保护，部署验收未通过。** 基线使用仿真真实状态、保留同样目标限幅时完成了协议，说明状态估计链是实际部署阻塞；这不证明限幅对步态质量没有影响，也不等于真机可用。不能通过提高保护阈值、偷偷改用仿真真值、每步移动参考原点来宣称解决。

未执行机器人通信或真机试验，未将 loop 部署门改为通过，训练 loop 保持暂停。停止响应时间仍需完整部署闭环或真机数据单独验收，本次不从未完成回放给出延时结论。

## 构建和人工操作入口

沿用现有 Linux 构建脚本和已安装 Unitree SDK。构建不会启动控制。

```bash
cd ~/G1_deploy
git switch yanni-G1-walk-beyondmimic
git pull --ff-only origin yanni-G1-walk-beyondmimic
python3 tools/check_beyondmimic_walk_package.py
bash tools/build_g1_deploy.sh
./deploy/robots/g1_29dof/build/g1_ctrl --help
```

以下配置入口供部署阻塞解决后的人工验证使用，当前不作为已通过真机验收的启动建议。实际网卡替换 `<G1网卡>`，必须显式给定 `--network`。

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
# 基线
./build/g1_ctrl --network <G1网卡> --domain 0 --config config/beyondmimic_walk.yaml --log
# 候选：先退出上一个控制器
./build/g1_ctrl --network <G1网卡> --domain 0 --config config/beyondmimic_walk_contact.yaml --log
```

保留起始关节误差、倾斜、anchor 误差、LowState/输出超时、关节范围、目标变化速率和 PD 力矩包络保护，新增无线遥控器超时检查。手柄缓存和低状态快照不跨线程无锁访问。

CSV 写入 `deploy/robots/g1_29dof/log/beyondmimic_walk/<model_id>_<timestamp>.csv`，包括指令、相位、参考、估计状态、动作、SDK 目标和限幅数量。`estimated_*` 列不是实际位置或速度。

## 开发者复现

包导入仅开发机需要现有 Torch/Numpy/PyYAML：

```bash
python tools/import_beyondmimic_walk_models.py --archive <冻结双模型归档目录>
python3 tools/check_beyondmimic_walk_package.py
```

C++ 探针只依赖 Eigen，与真机共享参考、观测、里程计和限幅实现：

```bash
cmake -S deploy/robots/g1_29dof -B deploy/robots/g1_29dof/build-probe -DBUILD_ROBOT_CONTROLLER=OFF -DBUILD_BEYONDMIMIC_OFFLINE=OFF
cmake --build deploy/robots/g1_29dof/build-probe --target beyondmimic_walk_probe
python tools/verify_beyondmimic_walk.py --package <模型包> --prepare <fixture目录>
python tools/verify_beyondmimic_walk.py --package <模型包> --checkpoint <PT> --runtime-dir <归档reference_runtime> --onnx-parity
python tools/verify_beyondmimic_walk.py --package <模型包> --fixture <fixture目录> --probe <探针程序> --checkpoint <PT> --runtime-dir <归档reference_runtime> --protocol docs/validation/walk/turn_support_0.json --mjcf <训练G1.xml> --output <离线结果目录>
```

物理验证复用现有 gmr/MuJoCo，未安装新环境。`--truth-diagnostic`、`--raw-targets-diagnostic` 只属于离线验证工具，真机状态没有这些绕过开关。

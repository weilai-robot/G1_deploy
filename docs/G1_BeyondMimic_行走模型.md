# BeyondMimic G1-29DOF 双模型

## 当前选择

| 模型包 | 实际训练来源 | 配置 | 用户反馈 |
|---|---|---|---|
| `walk_baseline` / `stable_5997_r26` | r26，TASK_20261102_078，model_5997.pt | `beyondmimic_walk.yaml` | 脚步更稳，流畅度较低 |
| `walk_contact` / `fluid_7996_r29` | r29，TASK_20261103_048，model_7996.pt | `beyondmimic_walk_contact.yaml` | 步态更流畅，脚步稳定性较低 |

`contact` 是保留的配置文件名，现在明确对应本轮7996。不要使用被否定的 TASK_20261102_155/model7996，也不要混用旧160维的5997。两个包均包含 ONNX、原生周期步态、精确控制参数和来源 manifest。完整 checkpoint SHA、任务、源码 commit 和配置校验见 [双模型清单](../deploy/robots/g1_29dof/config/policy/beyondmimic/walk_models.json)。

## 观测和控制

actor154由参考关节位置/速度、参考相对躯干朝向、IMU角速度、编码器关节位置/速度、上一动作组成，输出29关节动作。内置训练归一化保留。该模式不运行腿式里程计，不接收仿真位置/线速度，也不执行依赖位置估计的累计 anchor-error 检查；它不提供全局定位或全局路径偏离检测。

保留起始姿态、实测关节范围、倾斜、通信/手柄超时、推理输出超时、目标速度和 PD 包络保护。新增包加载检查要求原生训练目标契约与实际控制参数一致，FSM最大目标速度也须一致；不通过时拒绝该配置，不通过放宽限制改善视频。FixStand关节姿态和SDK顺序PD来自对应模型初始支撑帧。

## 手柄

| 操作 | 行为 |
|---|---|
| L2 + 上 | FixStand，过渡到当前模型初始支撑姿态 |
| FixStand完成、摇杆回中后 R1 + X | 进入 BeyondMimicWalk |
| 左杆前推 | 训练范围内的前向速度 |
| 前推时拨右杆 | 行走中转向 |
| 左杆回中 | 按支撑相制动，继续策略站立 |
| 再次前推 | 从当前支撑相起步 |
| L2 + B 或 Ctrl+C | 退出策略并发布阻尼 |

侧移、倒退和原地转向不受支持。回中与 Passive 阻尼是不同操作。参考周期回绕和指令切换不重置机器人状态或动作历史。换模型需退出控制器后选择另一份配置，不支持热切换。

## 实际验证状态

[验证汇总](beyondmimic_walk_validation.json) 及 [本次证据](validation/selected_pair_20261103/) 区分构建、接口、离线功能和真机验收。原冻结协议结果来自模型实际连续状态CSV，不能仅凭训练完成、存活或视频观感判断过门。

| 项目 | 5997 | 本轮7996 |
|---|---|---|
| 原七组完整回放 | 7/7完成 | 7/7完成 |
| 原功能门 | 7/7通过 | 2/7通过，部分初始站立/停车速度或偏航未通过 |
| 原接触质量门 | 3/4通过，右侧RMS目标未通过 | 3/4通过，左侧RMS非劣门未通过 |
| 用户选择 | 保留稳定版 | 保留流畅版作比较 |
| 真机数据 | 没有 | 没有 |

停止响应时间在各模型的 `evaluation-summary.json:runs[].stopResponses` 单独记录，没有用户指定的硬截止；`null` 表示该片段没有观察到满足原稳定门的持续尾段。7996的留存不撤销未过门记录。此前旧160模型的腿式里程计失败属于历史路径，不能继续当作当前154模型的阻塞原因。

## 构建和离线配置检查

```bash
cd ~/G1_deploy
python3 tools/check_beyondmimic_walk_package.py
bash tools/build_g1_deploy.sh
./deploy/robots/g1_29dof/build/g1_ctrl --help
./deploy/robots/g1_29dof/build/beyondmimic_offline --check-walk-config deploy/robots/g1_29dof/config/beyondmimic_walk.yaml
./deploy/robots/g1_29dof/build/beyondmimic_offline --check-walk-config deploy/robots/g1_29dof/config/beyondmimic_walk_contact.yaml
```

离线检查加载实际 ONNX、原生目标契约、步态和模型专用 FixStand，不建立 DDS 连接。Linux x86_64 实际构建结果不能当作 ARM64 已实测；机载机按自身架构构建。

人工试验选配置：

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
# 5997稳定版；填写实际网卡
./build/g1_ctrl --network <G1网卡> --domain 0 --config config/beyondmimic_walk.yaml --log
# 先退出上个进程，再选择7996流畅版
./build/g1_ctrl --network <G1网卡> --domain 0 --config config/beyondmimic_walk_contact.yaml --log
```

现场操作由用户执行。本次未连接或控制真机。日志在 `log/beyondmimic_walk/<model_id>_<timestamp>.csv`；154模式不可用的 `estimated_*` 与 anchor 估计列留空，不编造零值。

## 开发者维护

用户选择的完整PT/配套runtime及历史旧模型保留在loop归档。更新双模型时使用带SHA的 `pair.json` 归档：

```bash
python tools/import_beyondmimic_walk_models.py --archive <已冻结双模型归档> --replace
python3 tools/check_beyondmimic_walk_package.py
```

导入器先验证两个实际checkpoint、runtime源码、154观测及原生目标契约，再暂存导出；显式 `--replace` 才替换已有两个入口，并在归档中备份被替换包。机器人运行时无需这些PT和开发依赖。

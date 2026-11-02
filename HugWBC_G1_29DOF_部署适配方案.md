# G1 29DOF 部署代码适配 HugWBC：R41 / R42 实施方案

编写日期：2026-11-02。本文交给后续执行 AI 实现；本次只切换分支、核对代码和模型并写本文，没有修改部署代码、导出 ONNX、运行机器人或恢复训练。

## 1. 实施基线与范围

- 部署仓库：`D:/工作资料/02机器人/01F1项目/1_G1_deploy/`。
- 远端：`https://github.com/weilai-robot/G1_deploy.git`。
- **实际目标分支是 `yanni-G1-walk-hugWBC`**，不是 `yanni-G1-hugWBC`；已创建本地跟踪分支并切换。
- 本文核对的部署提交：`8807bc209761bd005c331681a884a0fed212a761`；远端该分支与 `main` 同提交。开始实现前再检查工作区，保留用户随后产生的改动。
- **用户已明确硬件只有 G1 29DOF。使用现有 `deploy/robots/g1_29dof` 可执行程序和 DDS 类型，不转到 `g1_23dof` 目录，不削掉腰部自由度。**
- 实现两套可选择的部署包：R41 腿腰策略＋参考双臂，以及 R42 在相同参考双臂下适配续训的策略。默认选择用户接受的 R41，R42独立保留供对比；禁止根据文件名排序自动挑“最新模型”。
- 保持既有 Velocity、Mimic、BeyondMimic 功能。新增独立 HugWBC 状态及配置，不把原有状态的观测或关节映射全局改成 HugWBC。
- 本任务不继续训练、不精进屈膝。训练 loop 已暂停，见[暂停记录](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/PAUSED.md)。

## 2. 实际模型、控制器与证据位置

以下数值来自已保存的 manifest、配置字段、训练源码以及实际 `.pt` 内张量形状；不是按常见 RL 部署模板估算。

| 版本 | 模型与定位 | 必须使用的控制方式 |
|---|---|---|
| R41 | `D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/accepted/r41-reference-arms-20261030/model_12600_g1r41.pt`；用户接受的视频版本 | 腿腰使用策略输出；完整覆盖双臂输出；手臂使用接受版本的 G1 PD。**不是直接运行原 R41 纯策略或原弱手臂 PD** |
| R42 | `D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/experiments/exp-20261030-004/model_13200_g1r42.pt`；固定参考双臂下完成适配的候选 | 同一套参考、手臂 PD、动作历史和命令标志 |

模型原件 SHA256：

```text
R41  2e468ff18b5d34797530182c1e87f0e09441ee11aadd24661fff10c1042f6b51
R42  4933371459e0e26ec74833c4161be6774201c7158d6810334578e89bec37cb04
```

来源与恢复入口：

| 标记 | 实际文件、应核对的字段或函数 |
|---|---|
| A | [两版保存索引](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/preserved-r41-r42.json)：`R41`、`R42`、模型 SHA、暂停状态 |
| B | [R41 manifest](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/accepted/r41-reference-arms-20261030/manifest.json)：`control`；[原视频执行封装](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/accepted/r41-reference-arms-20261030/audit_reference_execution.py)：`calculate_action`；[实际执行范围](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/accepted/r41-reference-arms-20261030/assistance-scope.json) |
| C | [R42 固定控制器快照](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/experiments/exp-20261030-004/controller-snapshot/g1r42.py)：`arm_reference_target`、`calculate_action`；[配置快照](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/experiments/exp-20261030-004/controller-snapshot/g1r42_config.py) |
| D | [G1 训练配置](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/x1-training/legged_gym/envs/g1/g1_config.py)：`init_state.default_joint_angles`、`control`；[G1Interrupt 配置](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/x1-training/legged_gym/envs/g1/g1interrupt_config.py)：腿腰 PD、`policy.NetModel.MlpAdaptModel` |
| E | [H1 公共执行代码](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/x1-training/legged_gym/envs/h1/h1.py)：`step`、`post_physics_step`、`_preprocess_obs`、`compute_observations`、`_step_contact_targets`、`_compute_torques`、`reset`；[干预公共代码](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/x1-training/legged_gym/envs/h1/h1interrupt.py)：`calculate_action`、`_preprocess_obs` |
| F | [模型结构](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/x1-training/rsl_rl/rsl_rl/modules/net_model.py)：`BaseAdaptModel.forward`、`MlpAdaptModel.memory_encoder`；[ActorCritic](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/x1-training/rsl_rl/rsl_rl/modules/actor_critic.py)：`act_inference` |
| G | [基础数值配置](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/x1-training/legged_gym/envs/base/legged_robot_config.py)：`normalization`、`sim.dt`；[H1 配置](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/x1-training/legged_gym/envs/h1/h1_config.py)：`env.include_history_steps`、步态命令范围 |
| H | [历史缓存](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/x1-training/legged_gym/legged_utils/observation_buffer.py)：`reset`、`insert`、`get_obs_tensor_3D` |
| I | [训练资产 URDF](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/x1-training/resources/robots/g1/g1_29dof_anneal_23dof.urdf)：关节类型、轴、限位、腰部及 IMU 链；[实际零更新 CSV](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/experiments/exp-20261030-004/baseline-hybrid/seed1/rollout.csv)：`q_*` 列顺序、`cmd_*` |
| J | [R42 评估器快照](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/experiments/exp-20261030-004/controller-snapshot/eval_upright.py)：`_resample_commands`、原生混合控制协议；[R42 对照结果](D:/工作资料/02机器人/01F1项目/oma-runs/hugwbc-g1/.oma/experiments/exp-20261030-004/adaptation-progress.json) |

原训练仓库分支 `hugwbc-g1`：R41 原提交 `b219b94b433d077fca673c945c22f67ab03ff3f3`，R42 原提交 `4c1234ba91a71143cf6f85eb2e33d139f15b36d0`。源文件未来若变动，以快照、模型 SHA 和相应提交恢复，不能用其他训练线的同名 `model_*.pt`。

## 3. 推荐代码结构与逐文件修改清单

下表路径相对于部署仓库根 `D:/工作资料/02机器人/01F1项目/1_G1_deploy/`；标注“新增”的文件目前尚未实现。

| 文件 | 修改内容 |
|---|---|
| `tools/export_hugwbc.py`（新增） | 从 R41/R42 checkpoint 重建**完整 actor**并分别导出 ONNX；生成配置、manifest、原模型 SHA、关节名、历史与缩放契约；不得套用 BeyondMimic 导出器 |
| `tools/check_hugwbc_package.py`（新增） | 校验两版包的文件哈希、形状、23→29 映射、参考控制、PD、模型身份；检查不依赖 Isaac Gym 或 DDS |
| `tools/verify_hugwbc.py`（新增） | 以真实零更新导出的 fixture 比对 PyTorch、Python ORT、C++ core/ORT；覆盖时钟、历史和参考动作时序 |
| `deploy/include/hugwbc/core.h`（新增） | 纯 C++ 数值核心：SDK 状态映射、pelvis 观测、历史、命令、步态时钟、参考双臂、动作缩放、腕部保持；不依赖 DDS，供离线和真机共用 |
| `deploy/include/hugwbc/runner.h`、`package.h`（新增） | ORT 推理、严格形状/数据类型检查、配置及 manifest 加载、版本身份验证 |
| `deploy/robots/g1_29dof/include/State_HugWBC.h`、`src/State_HugWBC.cpp`（新增） | 独立 FSM 状态；快照/推理线程/完整输出包/退出逻辑/日志；通过配置选 R41 或 R42 |
| `deploy/robots/g1_29dof/config/hugwbc_r41.yaml`、`hugwbc_r42.yaml`（新增） | 独立 FSM 配置、模型目录、固定前行测试命令、起步过渡和故障处理；不覆盖原 `config.yaml` |
| `deploy/robots/g1_29dof/config/policy/hugwbc/r41/`、`r42/`（新增） | 各自的 `policy.onnx`、`deploy.yaml`、`manifest.json`，不能共享一个可覆盖的 policy 文件 |
| `deploy/tools/hugwbc_offline.cpp`（新增） | 无 DDS 的数值回放程序，可在未连接机器人时校验模型和核心；同时作为后续闭环仿真的控制核心 |
| `deploy/robots/g1_29dof/main.cpp` | 注册并按显式 HugWBC 配置启用新状态，打印实际模型身份；沿用现有 29DOF 类型核对、退出与 DDS 通道占用检查 |
| `deploy/robots/g1_29dof/CMakeLists.txt` | 新增 HugWBC runtime/offline 目标并链接既有 Eigen/yaml-cpp/ORT；保留原目标，硬件程序仍使用 Linux Unitree SDK |
| `tools/build_g1_deploy.sh` | 当前脚本无条件执行 `check_beyondmimic_package.py`；改为明确选择包检查器或增独立 HugWBC 构建脚本，避免构建 HugWBC 反而只检查 BeyondMimic |
| `docs/G1_HugWBC_部署流程.md`（后续实现时新增） | 记录真实导出/构建/离线校验结果、模型选择、接口、硬件帧标定、启停/转向的验证范围与日志位置 |

复用方式：现有 [State_BeyondMimic.cpp](D:/工作资料/02机器人/01F1项目/1_G1_deploy/deploy/robots/g1_29dof/src/State_BeyondMimic.cpp) 的快照、锁、异常退出和目标包交换方式可参考；不能复用它的动作/观测语义、轨迹表、里程计或有限片段结束逻辑。现有 [State_RLBase.h](D:/工作资料/02机器人/01F1项目/1_G1_deploy/deploy/include/FSM/State_RLBase.h) 的通用观测与动作管理器也不能未经契约比对直接复用。

## 4. Actor 导出：不是普通单帧 MLP，也不是依赖真值的 teacher

来源 F、D、H，以及本次对两个实际 `.pt` 中 `model_state_dict` 张量形状的只读核对：

| 项目 | 实际契约 |
|---|---|
| 模型类型 | `MlpAdaptModel`；`is_recurrent=False`，外部维护历史，无 LSTM/GRU hidden state |
| Actor 输入 | float32，推荐固定形状 **`[1,5,88]`**，历史从旧到新；若另选 `[1,440]`，必须只在 ONNX 内显式 reshape，不能更改排列 |
| Actor 输出 | float32，固定 **`[1,23]`**，确定性均值动作；不采样 Gaussian，不使用 `std` |
| 每帧本体观测 | 前 75 维 |
| 最新命令＋时钟 | 最新帧 `[75:88]`，共 13 维 |
| 记忆编码器 | `actor.mem_encoder`：375→256→128→32，其中375=5×75 |
| 状态估计器 | `actor.state_estimator`：32→64→32→3 |
| 动作网络 | `actor.low_level_net`：123→256→128→32→23，其中123=32+3+75+13 |
| 激活 | ELU，中间层使用；最后输出无激活截断 |

准确 forward：

```python
pro_history = obs_history[..., :75]                 # [1,5,75]
latent = mem_encoder(pro_history.flatten(-2, -1))   # [1,32]
estimated_priv = state_estimator(latent)            # [1,3]
latest_pro = obs_history[..., -1, :75]
latest_cmd_clock = obs_history[..., -1, 75:88]
raw_actions = low_level_net(cat(
    [latent, estimated_priv, latest_pro, latest_cmd_clock], dim=-1))
```

- 必须完整导出 `mem_encoder`、`state_estimator`、`low_level_net`，严格加载对应 actor 前缀的权重与 bias。不能只导出 `low_level_net`，不能把前五帧顺序反过来。
- 推理时 `privileged_obs=None`、`sync_update=False`。真机**不需要输入仿真接触力、地形扫描、真实根线速度或外部动捕位置**；状态估计器的预测直接在图内使用，不另行反缩放后再喂动作网络。
- 训练/critic 的完整观测维度不是部署 actor 的输入维度。不能把 critic 输入或整份观测用零填充后冒充部署输入。
- 该 actor 没有 BeyondMimic 的 empirical observation normalizer。按下一节的显式缩放输入，不添加未经训练的归一化层。
- 可在导出脚本中用纯 PyTorch 重建这些 Linear/ELU 层，以避免部署导出依赖 Isaac Gym；但必须与训练类的确定性推理逐项校验，不能仅凭层形状一致通过。
- 保持部署端 ONNX Runtime 的现有版本/架构选择；当前 CMake 和构建脚本使用的版本来自其 `ONNXRUNTIME_ROOT` / `runtime_version`，不要为本适配顺便全局升级依赖。ONNX opset、导出工具版本和误差实测值写进各自 manifest。
- 本文没有执行 ONNX 导出，因此**没有已验证的 ONNX 文件或导出误差数值**。执行 AI 必须实际生成并检查，不能在报告里声称这些检查已经通过。

## 5. 关节顺序、29DOF 映射和双腕固定

来源：I 中实际 CSV 的 `q_*` 列顺序、D 的零动作默认角，以及现有部署 [export_beyondmimic.py](D:/工作资料/02机器人/01F1项目/1_G1_deploy/tools/export_beyondmimic.py) 的 `SDK_NAMES`。SDK 名称与编号另可核对 [Unitree 官方 G1 低层示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py) 的 `G1JointIndex`。

| 模型索引 | 训练关节名 | SDK 电机索引 | 零动作默认角 q0（rad） | 实际目标来源 |
|---:|---|---:|---:|---|
| 0 | left_hip_pitch_joint | 0 | -0.1 | 策略 |
| 1 | left_hip_roll_joint | 1 | 0 | 策略 |
| 2 | left_hip_yaw_joint | 2 | 0 | 策略 |
| 3 | left_knee_joint | 3 | 0.3 | 策略 |
| 4 | left_ankle_pitch_joint | 4 | -0.2 | 策略 |
| 5 | left_ankle_roll_joint | 5 | 0 | 策略 |
| 6 | right_hip_pitch_joint | 6 | -0.1 | 策略 |
| 7 | right_hip_roll_joint | 7 | 0 | 策略 |
| 8 | right_hip_yaw_joint | 8 | 0 | 策略 |
| 9 | right_knee_joint | 9 | 0.3 | 策略 |
| 10 | right_ankle_pitch_joint | 10 | -0.2 | 策略 |
| 11 | right_ankle_roll_joint | 11 | 0 | 策略 |
| 12 | waist_yaw_joint | 12 | 0 | 策略 |
| 13 | waist_roll_joint | 13 | 0 | 策略 |
| 14 | waist_pitch_joint | 14 | 0 | 策略 |
| 15 | left_shoulder_pitch_joint | 15 | 0 | 参考覆盖 |
| 16 | left_shoulder_roll_joint | 16 | 0 | 参考覆盖 |
| 17 | left_shoulder_yaw_joint | 17 | 0 | 参考覆盖 |
| 18 | left_elbow_joint | 18 | 0 | 参考覆盖 |
| 19 | right_shoulder_pitch_joint | 22 | 0 | 参考覆盖 |
| 20 | right_shoulder_roll_joint | 23 | 0 | 参考覆盖 |
| 21 | right_shoulder_yaw_joint | 24 | 0 | 参考覆盖 |
| 22 | right_elbow_joint | 25 | 0 | 参考覆盖 |

映射数组：

```text
model_to_sdk = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,22,23,24,25]
```

**不要复制现有 Velocity 的交错排列，也不要把模型的19号关节写到SDK的19号腕关节。** 读回 q/dq 和写出目标必须使用同一名称映射。

硬件29DOF与模型23维之间的差额来自双腕：

| 额外硬件关节 | SDK 索引 | HugWBC 下的处理 |
|---|---|---|
| left_wrist_roll / pitch / yaw | 19 / 20 / 21 | 固定位置保持，不进入23维模型观测/动作 |
| right_wrist_roll / pitch / yaw | 26 / 27 / 28 | 同上 |

I 中这**六个**腕关节都是 `type="fixed"`，固定 origin 的 `rpy="0 0 0"`。所以本方案采用双腕六轴在相应 SDK 零位的固定保持，不能只固定两侧 pitch/yaw 共四轴后，让 wrist roll 继续自由运动或沿用另一模型的动作。

这是**软件位置保持**：仍向29个电机下发完整命令，保持现有29DOF `mode_machine` 检查，不尝试把机器人切成另一硬件型号。名义 `q_wrist=0` 来自训练资产；实施时核对编码器零位与该资产的一致性。固定关节的仿真刚性不等于真机有限 PD，必须记录实际腕角/误差。

**腕部 PD 没有训练值。** 当前 HugWBC 没有腕部驱动器参数。可复用本分支现有 `FixStand` 的腕部保持实现作为起点，明确它是部署保持参数，不冒充已训练 PD；若需改值，单独说明来源和验证。不要从模型对不存在的腕输出凑数。

## 6. 每帧88维观测：精确切片与缩放

来源 E 的 `_preprocess_obs`，G 的 `normalization.obs_scales`，D 的命令维度。按**先缩放、后整帧 clip 到 [-100,100]**，再插入历史。真机不叠加训练随机噪声或人工随机时延。

| 切片，左闭右开 | 内容 | 缩放 |
|---|---|---|
| `[0:3]` | pelvis 本体系角速度 | ×0.25 |
| `[3:6]` | pelvis 坐标系内的单位重力方向 | 不额外缩放 |
| `[6:29]` | 训练顺序 `(q23 - q0)` | ×1.0 |
| `[29:52]` | 训练顺序 `dq23` | ×0.05 |
| `[52:75]` | 上一次提交的**混合动作**，23维 | 不乘动作尺度 |
| `[75:86]` | 以下11维命令，逐维缩放 | 见下表 |
| `[86:88]` | 左右脚步态时钟 | 两个 sin，不是 sin/cos 对 |

来源 J 的固定评估命令及 I 的原始 `cmd_*`；命令缩放来源 G 和 E/H1Interrupt 初始化：

| cmd 索引 | 意义 | 已验证直行原始值 | 输入缩放 |
|---:|---|---:|---:|
| 0 | 前向速度 vx，m/s | 0.5 | 2.0 |
| 1 | 侧向速度 vy，m/s | 0 | 2.0 |
| 2 | 偏航角速度 wz，rad/s | 0 | 0.25 |
| 3 | 步态频率 | 1.5 | 1.0 |
| 4 | 两脚相位偏移 | 0.5 | 1.0 |
| 5 | 支撑比例 | 0.5 | 1.0 |
| 6 | 足抬高命令 | 配置字面值0.1；CSV为float32记录值 | 0.15 |
| 7 | 身高相对命令 | 0 | 2.0 |
| 8 | 身体pitch命令 | 0 | 0.5 |
| 9 | 遗留名 `waist_roll` 的腰控制命令 | 0 | 0.5 |
| 10 | interrupt 标志 | **0** | 1.0 |

`cmd9` 当前固定0，不扩展新腰控制语义；若以后开放，先查真实继承的 `_reward_waist_control`，不能按变量名猜控制轴。

**双臂虽然来自外部参考，interrupt仍必须为0。** 这就是接受视频和R42原生训练的契约，不能为了表达“有外部控制”自行改为1。不要启用 `disturb_masks` 或 `interrupt_mask`；辅助控制通过状态名和包 metadata 披露。

### 6.1 历史缓存与初始化

- H：历史由旧到新排列；每个策略tick只插入一帧，不能在FSM发送循环重复插帧。
- 缓存共5帧，每帧88维。actor只用历史帧的前75维，以及最新帧的最后13维；仍建议完整按训练形状传入。
- H 的 `zero_pad=False` 表示 reset 时重复**传入的** `new_obs`。但是 E 的 `H1Robot.reset()` 是先从当时 `obs_buf` 填缓存，再执行一个零输入物理步。**不能据此宣称第一个策略输入必然是五帧最新传感器值，也不能把初始动作历史随意写成参考动作。**
- 执行 AI 应抓取真实 reset→第一推理→后续推理的 fixture，以它定义数值复现的启动契约。部署端从 FixStand 进入而非仿真物理重置，是单独的起步适配；若采用预热观测、插值或重新初始化缓存，必须显式记录并在闭环仿真验证，不能隐含在通用 manager 的 reset 内。
- 普通命令变化、相位回绕不重置历史；重新进入状态/异常恢复时，才执行明确的完整 reset 流程。

## 7. pelvis IMU 与时钟必须正确

### 7.1 输入的是骨盆，不是奖励中选用的 torso

来源 E 的 `post_physics_step`：角速度与 projected gravity 均由 `root_states` 的 pelvis 姿态计算。R41/R42 把直立奖励扩大到 pelvis/torso，不改变 actor 的根坐标系。

现有 [unitree_articulation.h](D:/工作资料/02机器人/01F1项目/1_G1_deploy/deploy/include/unitree_articulation.h) 直接读取 `msg.imu_state()`，但没有在 HugWBC 契约中验证数据所属刚体。I 中训练URDF的 `imu_link` 又挂在 `torso_link`；**URDF里的IMU位置不等于当前 actor 用的根姿态来源。**

- 优先使用已确认属于 pelvis 的 IMU 姿态和角速度；核对所用 DDS/仿真桥的实际来源与外参，不只看字段名。
- SDK的四元数按当前接入层的 `wxyz` 约定读取；训练 Isaac Gym 四元数内部为 `xyzw`，转换时必须显式处理。
- 构造 `projected_gravity = R_WP.transpose() * [0,0,-1]`，不喂带重力/运动加速度的加速度计原值。
- 如果拿到的是 torso IMU，使用实际三轴腰关节 FK 和角速度补偿恢复 pelvis。I 的腰链旋转为 yaw→roll→pitch；不能套只有 waist_yaw 的 H1 变换，也不能把 torso gyro仅旋转而不减相对腰部角速度。
- 静止/仅动腰的记录应能证明 pelvis 观测不错误地随 torso 一起转动。无这项验证，不能声称观测帧已适配。

外部核对来源：[Unitree MuJoCo 官方仓库](https://github.com/unitreerobotics/unitree_mujoco#supported-unitree-sdk2-messages)区分 G1 的 torso `rt/secondary_imu`，也说明真实低层控制时不能依赖仿真仍提供的 `SportModeState`。因此不要把仿真桥能提供根真值线速度作为真机 actor 的必要输入。具体机器人/SDK版本的传感器映射仍需实施时核对。

### 7.2 步态时钟

来源 E 的 `_step_contact_targets`；基础策略周期由 G 的 `sim.dt=0.005` 与 D 的 `decimation=4` 得到 **0.02s，即50Hz**。部署的 [CtrlFSM.h](D:/工作资料/02机器人/01F1项目/1_G1_deploy/deploy/include/FSM/CtrlFSM.h) 已有 `dt=0.001` 的发送循环，两者不能混为一个频率。

```text
g <- remainder(g + policy_dt * cmd3, 1)
foot_phase_left  = remainder(g + cmd4, 1)
foot_phase_right = remainder(g, 1)
```

存一份上述**未按支撑比例重映射**的左右脚相位，供参考摆臂使用。然后为观测时钟按源码做支撑/摆动段重映射：对每侧相位u和支撑比例d，u<d时用 `u*0.5/d`，u>d时用 `0.5+(u-d)*0.5/(1-d)`，严格匹配源码的边界处理；最后输出左右两侧的 sin。

- 当前验证条件d=0.5时，此重映射数值等同原相位，但仍保留代码区分，不能把摆臂相位换成重映射值。
- 当前两脚偏移为半周期，观测是两个相差半周期的sin。**禁止复用现有 `gait_phase` 的 sin/cos 实现。**
- 站立时：原实现的全局g仍推进，但脚相位/观测时钟置0，参考摆臂的周期项取消。不要擅自实现成停止全局时钟，或每次站走切换都从g=0重新起步。
- 同一个策略tick的观测时钟和肩参考使用同一份相位；不能一个更新前、另一个更新后，或在发命令时再推进一次时钟。

## 8. 双臂参考、动作历史与PD

来源 B/C：两版使用相同公式和PD。设A、δ来自接受manifest：

```text
A = 0.09178921571001411
delta = 0.30418359554562746
speed_scale = clamp(cmd_vx / 0.5, -1, 1)
walking_mask = not standing

left_shoulder_pitch_ref  = 0.3 - A*cos(2*pi*foot_phase_right - delta)*speed_scale*walking_mask
right_shoulder_pitch_ref = 0.3 - A*cos(2*pi*foot_phase_left  - delta)*speed_scale*walking_mask
left_shoulder_roll_ref   = +0.27
right_shoulder_roll_ref  = -0.27
both_shoulder_yaw_ref    = 0
both_elbow_ref           = 0.97
```

角度单位rad。参考是关节目标，不是手端笛卡尔轨迹，不需要在线IK或测量真实脚的前后位置；使用与策略一致的步态时钟。

### 8.1 正确的动作合成与历史

来源 E/H1Interrupt.calculate_action 与 B/C：

```text
a_raw = actor(history)                       # 23维，原始确定性输出
a_hybrid = a_raw
a_hybrid[arm_indices] = (q_arm_ref - q0_arm) / 0.25
a_clipped = clip(a_hybrid, -100, 100)
q_target23 = q0 + 0.25 * a_clipped

next_obs_action_history = a_hybrid           # 注意：不是a_raw，不是q_target23
```

- `self.actions` 在训练/接受执行中保存的是替换后的、clip之前的混合动作。观测整帧的[-100,100]剪裁仍在第6节进行。不要用另一部署模型的“原始网络输出历史”替代它。
- 参考臂的肘/roll已经构成最终目标，不能在 `.97` 或 `.27` 的 FixStand offset上再次相加。
- 名义q0中所有臂角仍是0，不能因为准备姿态肘为.97就把它改成观测中的默认肘角。
- 原执行 `reference_velocity_feedforward=false`；采用位置PD、目标速度0。不能默认附加参考角速度前馈或重力补偿后称为原视频控制器。

### 8.2 必须使用的名义 PD 和仿真力矩上限

来源：腿腰来自 D 的 `G1InterruptCfg.control`；参考双臂来自 B/C覆盖的 `G1Cfg.control`；力矩上限来自 D/G1Cfg 的 `control.torque_limits`。

| 关节组，双侧适用 | Kp | Kd | 训练/评估力矩饱和上限（Nm） |
|---|---:|---:|---:|
| hip pitch / roll / yaw | 100 | 2.5 | 88 |
| knee | 150 | 3 | 140 |
| ankle pitch / roll | 20 | 1 | 45 |
| waist yaw / roll / pitch | 150 | 3 | 88 |
| shoulder pitch | 90 | 2 | 25 |
| shoulder roll | 60 | 1 | 25 |
| shoulder yaw | 20 | 0.4 | 25 |
| elbow | 60 | 1 | 25 |

**不要把原G1Cfg整套腿腰PD恢复过来，也不要从R41本身的弱臂PD直接导出。** 当前接受R41使用了封装覆盖的强臂PD，R42原生采用相同覆盖；两者部署目标相同。

训练力矩计算是 `tau = clip(Kp*(q_target-q) - Kd*dq, ±tau_limit)`（确定性评估不加随机增益/电机offset）。这些上限是当前仿真配置，不是已查证的硬件额定值。

现有低层位置PD接口不能仅凭写入Kp/Kd/q就假定复现了该饱和。实施时核对固件限矩能力；如需在发送侧做保护，可根据当前q/dq将目标投影到 `q + (Kd/Kp)*dq ± tau_limit/Kp` 内，并与位置限位取交集。该等效关系仅对应当前测量时刻，必须验证实际发送与闭环表现，不能声称与仿真子步完全等价。额外目标速度限制、滤波、插值均作为显式部署包装记录，不暗中改动actor或历史契约。

PR/AB：训练的踝与腰是pitch/roll语义，接入前核对SDK串联PR模式，不能把这些目标当成并联电机A/B角。沿用29DOF型号检查；不根据模型23维切换机器人型号。

## 9. 控制周期、状态机和启动/停止范围

### 9.1 推荐单tick顺序

按E的“执行动作→物理状态更新→时钟/观测→下一推理”对应真实离散时序实现：

1. 取得同一LowState快照中的姿态、角速度、q/dq及时间戳，转为pelvis和训练关节顺序。
2. 推进本tick相位一次，形成对应新观测的脚相位与时钟；起始相位/首帧历史由真实reset fixture定义。
3. 用上一tick的 `a_hybrid`、当前命令和当前传感器拼88维，缩放/clip后压入历史一次。
4. 运行对应ONNX，保存23维 `a_raw`；用**刚才同一份相位**生成参考，合成 `a_hybrid`。
5. 得到23维目标，映射到29维SDK命令并补全六腕保持目标。将目标、PD、诊断信息、生成时间戳作为一个不可分割的包交给发送线程。
6. 在FSM已有发送周期中发布这份完整命令。它不再调用actor、不插入历史、不更新步态相位。
7. 保存本tick混合动作用于下一观测。策略线程超时不得通过连续补跑多次推理来“追赶”并把相位和历史跳过去。

建议复用现有BeyondMimic的互斥目标包/worker停止方式，而非让推理线程与FSM线程同时无锁访问LowCmd。包括PD在内的状态切换也必须保持一致；退出后旧worker不得再覆盖Passive命令。

### 9.2 准备姿态

现有 [29DOF config.yaml](D:/工作资料/02机器人/01F1项目/1_G1_deploy/deploy/robots/g1_29dof/config/config.yaml) 的FixStand含肩pitch=0、roll±0.25、肘.97、腕roll±.15；它不是本接受控制器的完整准备姿态。

HugWBC独立配置的准备姿态应围绕训练q0的腿腰、参考臂的中心姿态（pitch .3、roll±.27、yaw0、肘.97）与六腕0构造。借用现有FixStand插值机制，不做上电立即跳目标。**准备姿态与模型q0分开存储**，不能为便于站立而改写观测offset。

从该准备姿态进入策略，仍属于与仿真reset不同的起步包装，必须单独闭环验证。现有FixStand的插值时间、BeyondMimic的超时/入场阈值可以作为源码参考；没有数据支持的HugWBC阈值标为待实测，不写成训练已验证限值。

### 9.3 启停、转向与手柄

- 当前R41/R42接受视频与完整评估只验证固定前行。保留命令接口不等于可靠启停、倒退、侧移或转向已经通过。
- 初始实施先提供显式的固定前行测试profile，复现第6节命令；不要直接把现有Velocity的手柄三轴范围开放给HugWBC。
- 站立命令应同时设置 `vx/vy/wz=0`、`standing=true`、时钟输出0、取消参考周期项，保留臂中心姿态。不能仅把vx置零而仍按行走脚相位给策略输入。
- 若实现正常站停→起步、速度平滑或转向，单列为未验证扩展，先在仿真运行命令切换用例，记录状态/历史/相位。不能在README把它们直接写成已实现并验证的能力，也不能借此自动恢复训练loop。
- 紧急回Passive继续沿用既有故障处理，但**Passive阻尼不是正常的已训练制动/站停能力**。正常退出和异常退出分开实现及记录。

## 10. 必须完成的离线与闭环验收

### 10.1 先生成真实数值fixture

**没有：现有接受视频CSV并未保存完整88维历史、原始actor输出和全部控制中间量。** 不允许仅从CSV的关节角差分“估算”完整输入后宣称数值对齐。

执行AI在原训练代码上做**零优化更新**重放，分别抓取R41接受封装和R42原生控制器：

```text
checkpoint SHA / 源提交 / 环境与工具版本
reset、首个策略输入及历史缓存
SDK/训练顺序的q、dq；pelvis姿态与gyro
原始cmd11、缩放cmd11、standing标志
g、原始左右脚相位、重映射相位、clock2
观测88、历史[5,88]
raw_action23、hybrid_action23、clipped_action23
arm_reference8、q_target23、Kp/Kd/tau_limit
映射后的SDK29目标、wrist保持目标和实际输出限制状态
```

覆盖启动、跨相位回绕、正常行走，另覆盖零速度/站立与重新起步的扩展用例。若本机没有Isaac Gym，可使用既有仿真环境做只读模型重放；不使用冒烟训练后权重、不新增PPO更新。

### 10.2 数值对齐分层验收

1. 模型身份：SHA、实际iter、actor所有权重与预期形状严格一致；两版分别验证。
2. PyTorch原实现与纯PyTorch导出包装：对同一真实历史比较raw action；确认无critic/privileged真值输入。
3. ONNX Runtime：同fixture比较raw action，记录每层/输出的最大绝对误差和工具版本。**当前没有误差实测数据**，误差容限应经该对照测定并冻结；不为通过而放宽。
4. Python与C++ core：逐切片比较观测、历史、相位、臂参考、混合动作、目标及SDK映射。raw与hybrid分别保存，避免双臂覆盖掩盖actor导出错误。
5. 针对映射检查右臂和六腕，针对时序检查跨周期、站立和恢复；验证没有一tick相位偏移、历史倒序、重复乘scale或PD顺序错位。
6. 全部测试在无DDS连接时可执行；失败即具体报告哪一层、哪一维、哪个tick不一致。

### 10.3 C++控制闭环仿真

数值回放通过后，让同一C++ core/ORT在仿真中闭环运行；actor不能继续由Python替代。优先使用与I同语义的关节、惯量与固定腕模型；若改用完整29DOF模型，以六腕位置保持模拟固定腕，并明确资产/驱动器的差异，分别记录两版结果。

- 检查站高、pelvis/torso姿态、命令跟踪、存活、实际手端对称与对侧关系、肩pitch主导形态，以及腕保持误差。
- 沿用现有R41/R42冻结评估门和接受视频作对照，不把新增包装或另一物理引擎导致的差异归为“模型自然退化”后直接改权重。
- R42的已知结果是躯干变化率下降但速度RMSE上升，见J；不得为选择R42而追改原联合门或覆盖R41。
- IMU来自pelvis/torso两种输入路径分别验证；错误帧输入应能被测试发现，而不是仅做静止重力检查。
- 只有真正执行过的用例才能列“通过”。没有闭环结果就写“未验证”，不能把源码审查或ONNX输出正确代替仿真步态验证。

## 11. 日志与最终交付

实时日志至少包含：版本/模型SHA、状态与tick、传感器/推理/发布的单调时间戳、pelvis输入和其来源、q/dq、cmd11、standing、g/clock/脚相位、raw/hybrid动作、臂参考、SDK29目标、实际腕角、Kp/Kd、限制触发标志、推理耗时与退出原因。日志写入不能阻塞发送线程。

后续真机反馈重点是输入帧、时序、关节跟踪、参考臂实际运动、站停转换和保护触发，不以继续减少屈膝为目标。

执行AI交付时应给出：

- 仍在 `yanni-G1-walk-hugWBC` 分支的明确代码diff及提交；不把另一个BeyondMimic分支合进来作为本次适配。
- R41、R42各自的ONNX、配置、manifest、SHA和来源模型，硬件29目标/模型23输出/参考8臂/保持6腕的关系写清楚。
- 真实fixture、Python/ORT/C++对照结果、完整命令历史和闭环仿真视频；失败项和未验证项原样保留。
- 明确区分：固定前行复现、扩展启停/转向实现、以及用户实际真机验证，不能互相代替。
- 不重启训练loop，不覆盖原接受快照，不擅自更改摆臂公式、腿腰PD、动作offset或网络结构。

本次执行到“方案MD已生成、分支已切换”为止。后续代码实现、模型导出和部署验证均由用户指定的执行AI开展。

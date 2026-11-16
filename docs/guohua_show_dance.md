# guohua_show_dance 舞蹈接入

已将 `g1_dance/trained_models/guohua_show_dance.pt` 与用户指定的 `whole_body_tracking/motions/gLO_30fps_isaac.npz` 转换成独立部署包。动作文件内部为 50 Hz，共 2400 帧、48 秒。原 `dance_zui` 的模型包和默认选择保留。

**模型接口和 C++ 观测对照已通过；此前部署闭环未通过，已知存在失稳风险。** 2026-11-16 按用户真机调试要求，将运行段从 10 秒改为完整一次 48 秒；保留了全部部署保护，没有为了播放全长取消限幅或保护中止。

## 文件与接口

```text
deploy/robots/g1_29dof/config/beyondmimic_guohua_show_dance.yaml
deploy/robots/g1_29dof/config/policy/beyondmimic/guohua_show_dance/
  policy.onnx
  reference.bin
  deploy.yaml
  manifest.json
docs/guohua_show_dance_validation.json
```

| 项目 | 与 dance_zui 相同的定义 |
|---|---|
| ONNX 输入 | `obs`，float32，`[1,160]`，原始观测 |
| ONNX 输出 | `actions`，float32，`[1,29]` |
| 归一化 | 使用本检查点的 mean/std，已嵌入 ONNX，epsilon=0.01，不再次归一化 |
| 模型 | ELU MLP，160→512→256→128→29 |
| 周期 | 20 ms，50 Hz |
| 观测顺序 | 参考关节位置/速度、相对躯干位置/朝向、基座线/角速度、关节位置偏差/速度、上次原始动作 |
| 映射及控制参数 | 训练关节顺序、SDK 映射、默认角、动作缩放、训练 PD、运动学与原包一致 |
| 遥控操作 | L2+上进入 FixStand；稳定后 L2 持续 2 秒+右进入 BeyondMimic；L2+B 转阻尼 |

两个舞蹈复用 `State_BeyondMimic`，通过启动配置的 `policy_dir` 选择模型与参考动作；不需要新增控制器状态实现。新配置从第 0 帧运行到第 2399 帧，完整一次 48 秒，结束转阻尼。不会循环播放；保护触发时仍会提前退出，不能保证真机完成全段。

## 真机调试启动

同步代码后在机载 Linux 编译，以包含键盘线程退出修复和运行帧范围日志：

```bash
cd ~/G1_deploy
git pull --ff-only origin main
python3 tools/check_beyondmimic_package.py \
  deploy/robots/g1_29dof/config/policy/beyondmimic/guohua_show_dance
bash tools/build_g1_deploy.sh
```

保持防跌落吊装保护，按宇树流程进入开发模式后，在对应网卡上启动控制器（`eth0` 替换为实际机器人网卡）：

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
./build/g1_ctrl --network eth0 --domain 0 \
  --config config/beyondmimic_guohua_show_dance.yaml --log
```

检查 `FSM configuration` 指向新配置、`loaded` 显示 `2400 frames`，`segment` 显示 `start_frame=0 end_frame=2400 (exclusive), 48 s`。不指定 `--config` 仍为旧 `dance_zui`。

先按 L2+上，等待至少 3 秒，机器人进入原 `dance_zui` 的 FixStand 站姿（踝关节 150/3）。站稳后持续按住 L2 至少 2 秒，再按右；这时先进入首帧衔接阶段，不立即运行策略。完成衔接并通过实测检查后，才打印 `starting reference frame 0`，开始 48 秒舞蹈计时。L2+B 可在衔接和舞蹈阶段转阻尼。正常结束、保护中止或 Ctrl+C 退出都不会自动恢复站立，吊装需能承重。失稳、保护触发或线程异常时停止测试，不用连续重启或提高保护阈值强行继续。

## 首帧与站立参数

新动作第 0 帧为屈膝、不对称的动作姿态，关节角与原 FixStand 最大相差约 0.905 rad，参考关节速度最大约 3.46 rad/s，不能沿用旧站姿直接切入。按 2026-11-16 用户确认的流程，FixStand 保留原 `dance_zui` 的站姿和全部站立 PD，四个踝关节（SDK 4、5、10、11）为 Kp=150、Kd=3。首帧衔接在按舞蹈启动键后进行。

衔接已按用户要求缩短为 0.6 秒起的平滑关节插值，目标速度限制提高至 2 rad/s，距离较大时自动延长。按原站姿到首帧约 0.905 rad 的最大差值，姿态段实际最短约 0.68 秒；之后用 0.1 秒渐变到训练 PD，尽快交给策略控制。全过程仍按关节限位、目标变化速度和 PD 估算力矩共同约束输出，并保留 LowState 超时、目标更新超时、周期超时及骨盆倾斜退出检查。

训练 PD 渐变完成后，实测所有关节与首帧的最大角度偏差仍须不超过 0.25 rad，最大速度仍须不超过 0.5 rad/s；连续确认时间缩短为 0.04 秒，再重置策略历史/里程计并从第 0 帧推理。姿态和增益过渡结束后 0.5 秒内未到位则转阻尼，日志显示 `Preparation did not settle at motion first frame`。衔接时间不计入 48 秒动作时长。

以实际 29 关节参数进行理想反馈规划检查，约 0.82 秒允许进入策略；这不是动力学仿真或真机结果。实际衔接时间取决于起始姿态和反馈，不能保证更快一定更稳。新增快速衔接尚未经真机验证，也没有解决此前舞蹈闭环失稳问题。

参数为 `prepare_duration_s`、`prepare_gain_duration_s`、`prepare_settle_duration_s`、`prepare_settle_timeout_s`、`prepare_max_target_speed`、`prepare_max_joint_speed`。默认 `dance_zui` 未启用此阶段，保留原流程。该衔接是关节目标过渡，不是全身平衡控制；不能证明中间姿态或首帧可在真机静态站稳。原离线闭环验证从参考首帧直接初始化，没有覆盖新的衔接。

无硬件检查可运行 `python3 tools/test_beyondmimic_preparation.py`，覆盖姿态/增益顺序、速度和估算力矩约束、实测稳定门槛、超时及错误输入。规划器检查不等于完整控制器或真机稳定性验证；需在机载 Linux 重新编译。

## 离线验证

先按主文档在 Linux 编译 `beyondmimic_offline`，使用有 MuJoCo、NumPy、ONNX Runtime、SciPy、PyYAML 的 Python 环境执行：

```bash
python3 tools/check_beyondmimic_package.py \
  deploy/robots/g1_29dof/config/policy/beyondmimic/guohua_show_dance

python tools/verify_beyondmimic.py \
  --package deploy/robots/g1_29dof/config/policy/beyondmimic/guohua_show_dance \
  --offline-binary deploy/robots/g1_29dof/build/beyondmimic_offline \
  --mjcf /实际路径/g1_dance/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml \
  --seconds 48 --report /tmp/guohua_show_dance_verification.json
```

需要验证完整 DDS 状态机时，仿真域使用 `lo` / domain 1，配置选择 `--config config/beyondmimic_guohua_show_dance.yaml`，按键与原舞蹈一致。上面的真机命令是用户要求的调试入口，不代表全段部署已经通过验证。

2026-11-12 使用实际 C++ 核心及 MuJoCo 3.8.0 验证：

| 检查 | 结果 |
|---|---|
| 64 组检查点与 ONNX 推理 | 最大误差约 1.91e-6，通过 |
| 64 组 C++ 观测对照 | 最大误差约 3.58e-7，通过 |
| C++ 与 Python ONNX 动作对照 | 最大误差约 1.95e-6，通过 |
| 错误 IMU、错误 SDK 映射、截断参考表拒绝 | 通过 |
| 腿部里程计、正常部署限幅、请求 10 秒 | 4.36 秒失稳，未通过 |
| 仿真真实状态、正常部署限幅、请求 10 秒 | 3.50 秒失稳，未通过 |
| 仿真真实状态、绕过目标限幅的诊断 | 10 秒通过；仅用于定位，不能作为真机依据 |
| 原 dance_zui 10 秒回归 | 通过 |

对照说明部署限幅会影响该动作的稳定性；还需要验证状态估计和准备过程。原始诊断见 [验证记录](guohua_show_dance_validation.json)。没有修改真机限幅，也没有进行真机测试。

验证工具同时修正了参考动作起点不在世界 XY 原点时的对照坐标变换；该修改只影响独立观测对照，不改变控制器。

## 重新导出

在已安装 `tools/requirements-beyondmimic.txt` 依赖的开发环境执行：

```bash
python tools/export_beyondmimic.py \
  --training-root /实际路径/g1_dance \
  --checkpoint /实际路径/g1_dance/trained_models/guohua_show_dance.pt \
  --motion /实际路径/g1_dance/whole_body_tracking/motions/gLO_30fps_isaac.npz \
  --output deploy/robots/g1_29dof/config/policy/beyondmimic/guohua_show_dance
```

更换参考动作或起始帧后，应同步核对 FixStand 目标角并重新验证，不能只替换权重文件。

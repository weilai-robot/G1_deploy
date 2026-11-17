# guohua_show_dance 舞蹈接入

已将 `g1_dance/trained_models/guohua_show_dance.pt` 与用户指定的 `whole_body_tracking/motions/gLO_30fps_isaac.npz` 转换成独立部署包。动作文件内部为 50 Hz，共 2400 帧、48 秒。原 `dance_zui` 的模型包和默认选择保留。

**模型接口和 C++ 观测对照已通过；此前部署闭环未通过，已知存在失稳风险。** 2026-11-17 按用户要求统一两支正常舞蹈：FixStand 站稳后直接运行策略，默认完整播放一次；保留各配置已有的保护设置。

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

检查 `FSM configuration` 指向新配置、`loaded` 显示 `2400 frames`，`segment` 显示 `start_frame=0 end_frame=2400 (exclusive), 48.00 s`。不指定 `--config` 仍为旧 `dance_zui`。

先按 L2+上，等待至少 3 秒，机器人进入原 FixStand 站姿（踝关节 150/3）。站稳后持续按住 L2 至少 2 秒，再按右，立即切回训练 PD 并从第 0 帧运行策略。两支正常舞蹈都不先插值到首帧、不保持首帧、不等待首帧到位。L2+B 转阻尼；完整动作结束或保护触发也转阻尼，吊装需能承重。

## 统一流程与配置

两支正常舞蹈均设置 `direct_from_fixstand: true`、`play_full_motion: true`。启动时与 FixStand 目标关节角的最大偏差须不超过 0.25 rad；不以参考首帧作为启动姿态门槛。策略从第 0 帧开始，观测仍使用实测状态和原参考动作，目标仍受已有部署限幅约束。

| 舞蹈 | 配置 | 默认完整长度 |
|---|---|---|
| dance_zui | `config/beyondmimic.yaml` | 11384 帧，227.68 秒 |
| guohua_show_dance | `config/beyondmimic_guohua_show_dance.yaml` | 2400 帧，48 秒 |

启动日志应显示 `full_motion=true direct_from_fixstand=true preparation=false`。完整播放直接取参考表的帧数，避免手写秒数造成尾帧截断。若需短段，显式设置 `play_full_motion: false`，再给出 `duration_s`。默认完整一次，不自动循环。

两支舞的 FixStand 站姿和全部站立 PD 保持一致，踝关节为 150/3。正常 guohua 配置已移除 `prepare_*` 参数。独立的 `beyondmimic_guohua_first_frame.yaml` 仍仅用于 [高 Kp 首帧保持测试](guohua_first_frame_test.md)，不属于正常舞蹈流程，也不启动策略。

guohua 首帧为屈膝、不对称动作，关节角与 FixStand 最大相差约 0.905 rad，参考关节速度最大约 3.46 rad/s。直接启动按用户要求实现，并不等于已验证能从该站姿稳定接管；历史闭环测试也没有覆盖这一直接启动流程。完整动作和直接启动都需单独验证，不能通过取消其他保护来保证播完。

可运行 `python3 tools/test_beyondmimic_playback.py` 检查完整帧范围、SDK 映射及直接启动的站姿门槛；这不等于动力学或真机验证。本次 C++ 修改需在机载 Linux 重新编译。

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

更换参考动作或起始帧后，应重新验证从 FixStand 直接启动的观测、控制输出和稳定性；正常模式不再要求将 FixStand 目标改成该动作首帧。

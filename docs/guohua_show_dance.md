# guohua_show_dance 舞蹈接入

已将 `g1_dance/trained_models/guohua_show_dance.pt` 与用户指定的 `whole_body_tracking/motions/gLO_30fps_isaac.npz` 转换成独立部署包。动作文件内部为 50 Hz，共 2400 帧、48 秒。原 `dance_zui` 的模型包和默认选择保留。

**模型接口和 C++ 观测对照已通过；当前部署闭环未通过，不用于真机测试。** 保留了全部部署保护，没有为了通过仿真取消限幅。

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

两个舞蹈复用 `State_BeyondMimic`，通过启动配置的 `policy_dir` 选择模型与参考动作；不需要新增控制器状态实现。新配置从第 0 帧选取 10 秒试验段，结束转阻尼。

## 首帧与站立参数

新动作第 0 帧为屈膝、不对称的动作姿态，关节角与原 FixStand 最大相差约 0.905 rad，参考关节速度最大约 3.46 rad/s，不能沿用旧站姿直接切入。新配置的 FixStand 目标已按 SDK 顺序设置为本动作第 0 帧，仍使用独立站立 PD；进入舞蹈时恢复训练 PD。

这只保证准备目标与首帧关节角一致，不能证明它可静态站稳。现有离线验证从参考首帧直接初始化，没有验证 3 秒准备过程或增益切换。

## 离线验证

先按主文档在 Linux 编译 `beyondmimic_offline`，使用有 MuJoCo、NumPy、ONNX Runtime、SciPy、PyYAML 的 Python 环境执行：

```bash
python3 tools/check_beyondmimic_package.py \
  deploy/robots/g1_29dof/config/policy/beyondmimic/guohua_show_dance

python tools/verify_beyondmimic.py \
  --package deploy/robots/g1_29dof/config/policy/beyondmimic/guohua_show_dance \
  --offline-binary deploy/robots/g1_29dof/build/beyondmimic_offline \
  --mjcf /实际路径/g1_dance/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml \
  --seconds 10 --report /tmp/guohua_show_dance_verification.json
```

需要验证完整 DDS 状态机时，仿真域使用 `lo` / domain 1，配置选择 `--config config/beyondmimic_guohua_show_dance.yaml`，按键与原舞蹈一致。不要在真机 domain 0 下运行尚未通过闭环的配置。

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

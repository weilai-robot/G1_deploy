# BeyondMimic 可部署观测修正（2026-10-31）

新训练的 actor 只接收参考关节位置/速度、参考相对躯干朝向、IMU 角速度、关节位置/速度、上一动作。训练真实位置和线速度保留给 critic、奖励与评测。源配置与原生管理器审计见 `validation/proprio154/checkpoint-audit.json`；不把观测维度检查当作行走验收。

新包使用 `beyondmimic_g1_154_v1` / `velocity_cyclic_proprio_v1`，输入154、输出29；旧160模型和默认配置继续走原路径。新模式完全不更新腿式里程计，也不执行依赖该估计的累计位置误差保护。由此失去全局路径偏离保护，不能声称获得了全局定位能力。保留姿态、实测关节范围、通信/指令超时、推理截止、动作/目标/力矩保护。日志中不可用的位置/线速度估计留空。

原有 `walk_baseline`、`walk_contact` 两模型及其配置不改、不替换。新包的导出工具 `tools/export_beyondmimic_proprio.py` 拒绝合成诊断权重及覆盖已有模型目录；正式包保存到机器人 `config/policy/beyondmimic/` 下的独立目录，单独生成 FSM 配置。此提交不提供新的可运行候选包，不切换默认模型。

实际 GM 接入轮 `exp-20261031-001` / `TASK_20261031_111` 已自然结束，训练源码 `e827df7cbf8fdd2a584e5f5a8857a85c39ac47da`。验证证据在 `validation/proprio154/`：

- `checkpoint-audit.json`：实际新权重、原生 actor154/critic286 与源码指纹。
- `checkpoint-progress.json`：首末权重有限值、优化器和归一化计数实际增长。
- `cpp-inference.json`：真实新权重的 C++ ONNX 与 Torch 一致性。
- `smoke-replay.json` 与 `smoke-states.csv.gz`：共享部署 C++ 观测/目标限制加 MuJoCo 连续物理反馈；改变世界位置/线速度不改变 actor 输入，154模式拒绝真值状态包。
- `synthetic-interface-preflight.json`：训练前的合成接口夹具，仅接口诊断，绝非训练模型；与真实新权重证据分开。
- `final-build-report.json`：最终源码匹配、Linux SDK 完整编译及离线 --help、154/160包契约和错误项拒绝测试。

动态结果仍未通过：`smoke-replay.json` 的 `completed_seconds=0.24`，触发 `measured_joint_range_guard`；腰部 roll 实测越过模型下限，实际值和上下限均在 `joint_range_violations`。这是64更新接入模型，不能用于真机或替换已有基线。正式4096环境/2000更新的新训练已在 `TASK_20261031_114` 启动，行为结果尚未收取。完整指令回放、长行走、转向、启停、足端质量和视频未验收前，部署门保持关闭。

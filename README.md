# G1_deploy：G1 29DOF 的 11k 基线部署

当前默认运行训练仓库 `unitree_rl_lab_train` 的 **model_10999_factorygait.pt**，来源 TASK_20261020_069。ONNX 与对应参数已经放入仓库，无需在真机安装训练环境。

已完成生产 C++ Core/Runner 的 Windows 编译、与原始 PyTorch 的数值对照及 MuJoCo 闭环检查；**尚未在 Linux 编译 DDS 主控制器，尚未做真机测试**。目标机的构建脚本会自动运行不连接机器人的推理对照检查。

完整操作顺序、部署子分支拉取方式、仿真与真机启动、遥控器操作和日志说明见：

- [G1_11k_部署流程](docs/G1_11k_部署流程.md)
- [本次实际验证记录](docs/velocity11k_validation.json)

在部署 Linux 电脑拉取你实际推送的子分支后：

```bash
cd ~/G1_deploy
python3 tools/check_velocity11k_package.py
bash tools/build_g1_deploy.sh
```

完成构建和部署控制器仿真检查后，现场人员按部署流程进行受保护的真机验证。构建不会自动启动机器人。

默认配置为 `deploy/robots/g1_29dof/config/config.yaml`，仅启用 Passive、FixStand 和固定 11k 的 Velocity。策略目录明确指定为 `config/policy/velocity/11k`，不会自动选取名称靠后的其他策略。

旧的 BeyondMimic 模型与专用配置保留，使用说明见 [G1_BeyondMimic_部署流程](docs/G1_BeyondMimic_部署流程.md)；它不是本分支的默认入口。

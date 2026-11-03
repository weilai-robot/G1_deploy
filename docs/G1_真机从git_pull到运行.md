# G1：从更新仓库到选择行走模型

本文对应分支 `yanni-G1-walk-beyondmimic` 的5997稳定版和本轮7996流畅版，详细来源和未过门项见 [行走模型](G1_BeyondMimic_行走模型.md)。两者均未进行真机验收；7996仍有停车和初始站立未过门项。现场真机试验由用户安排，本次只完成代码、构建和离线验证。

## 更新与检查

在机载开发电脑的已有部署仓库执行：

```bash
cd ~/G1_deploy
git status --short
git switch yanni-G1-walk-beyondmimic
git pull --ff-only origin yanni-G1-walk-beyondmimic
python3 tools/check_beyondmimic_walk_package.py
bash tools/build_g1_deploy.sh
./deploy/robots/g1_29dof/build/g1_ctrl --help
ldd ./deploy/robots/g1_29dof/build/g1_ctrl
```

保留并处理本地修改后再切换，不使用强制重置。构建脚本沿用已有 Unitree SDK和Linux依赖；自定义SDK安装路径使用 `UNITREE_SDK_PREFIX`。运行库按当前CPU架构选择。模型已在仓库，不需要在机器人安装Isaac Lab、PyTorch或训练源码。

构建还会对两个配置执行 `beyondmimic_offline --check-walk-config`，核对ONNX输入、模型初始支撑姿态、PD和训练目标参数。构建、校验及 `--help` 不控制机器人。

## 人工选择模型

先确认现场状态与实际网卡，再选择一条命令：

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
# 5997稳定版
./build/g1_ctrl --network <G1网卡> --domain 0 --config config/beyondmimic_walk.yaml --log
# 7996流畅版；先退出另一个控制进程
./build/g1_ctrl --network <G1网卡> --domain 0 --config config/beyondmimic_walk_contact.yaml --log
```

确认日志模型ID分别为 `stable_5997_r26` 或 `fluid_7996_r29`。L2+上进入模型专用FixStand；完成过渡、摇杆回中后R1+X进入行走。左杆前推控制速度，行走时右杆转向，回中保持策略站立，再推起步。L2+B或Ctrl+C退出到阻尼。这里是持续手柄行走，不是旧dance_zui的有限片段播放。

模型切换通过退出进程并更换配置完成。日志写入 `deploy/robots/g1_29dof/log/beyondmimic_walk/`，模型ID区分两版；真机反馈请保留完整CSV和视频。

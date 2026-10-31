# G1 BeyondMimic 手柄行走与启停

分支 `yanni-G1-walk-beyondmimic` 接入 loop 保留的两个 G1-29DOF 模型，沿用现有 Unitree FSM 和 ONNX Runtime 框架。两个模型分别保存，未替换用户认可的基线。训练 loop 保持暂停。

| 模型 | 配置（相对 deploy/robots/g1_29dof） | 定位 |
|---|---|---|
| exp003 / model_3998 | `config/beyondmimic_walk.yaml` | 用户认可的基线 |
| exp004 / model_5997 | `config/beyondmimic_walk_contact.yaml` | 保留对比的候选 |

新增 `BeyondMimicWalk` 状态：手柄前推控制速度、右杆控制行走中的转向；回中后按支撑相制动并保持，再推杆重新起步。周期回绕和指令变化均不重置机器人状态或策略历史。侧移、倒退、原地转向不在这两个模型的训练范围内。

**当前为待验证部署版本：现有腿式里程计的闭环回放未通过，触发位置误差保护；尚不能作为连续行走真机验收版本。** 训练视频的认可不等于部署端状态估计通过。实际结果见[行走模型说明](docs/G1_BeyondMimic_行走模型.md)和[验证数据](docs/beyondmimic_walk_validation.json)。

更新和离线检查（不会启动机器人）：

```bash
cd ~/G1_deploy
git switch yanni-G1-walk-beyondmimic
git pull --ff-only origin yanni-G1-walk-beyondmimic
python3 tools/check_beyondmimic_walk_package.py
bash tools/build_g1_deploy.sh
./deploy/robots/g1_29dof/build/g1_ctrl --help
```

已有 Linux SDK 和依赖时沿用现有构建方式。真机推理不需要 Isaac Lab、PyTorch 或训练仓库。两个 ONNX、原生循环步态、控制参数、关节映射和来源校验均随仓库保存。

- [手柄操作、模型选择、接口及当前部署阻塞](docs/G1_BeyondMimic_行走模型.md)
- [本次适配设计与副作用边界](docs/beyondmimic_walk_adaptation.md)
- [旧 dance_zui 有限片段说明](docs/G1_BeyondMimic_部署流程.md)（配置 `beyondmimic.yaml`，与新增行走状态独立）

控制框架来自 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)，训练架构基于 [BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)。原始许可见 [LICENCE](LICENCE)。

# G1 BeyondMimic 手柄行走与启停

分支 `yanni-G1-walk-beyondmimic` 维护用户选定的两个 G1-29DOF 模型，使用现有 Unitree FSM、共享 C++ 控制映射和 ONNX Runtime。2026-11-03 替换之前的160维行走模型；历史权重仍保存在训练归档和 Git 历史中。

| 可选模型 | 配置（相对 deploy/robots/g1_29dof） | 用户对比反馈 |
|---|---|---|
| 5997 稳定版，`stable_5997_r26` | `config/beyondmimic_walk.yaml` | 脚步更稳，流畅度较低 |
| 7996 流畅版，`fluid_7996_r29` | `config/beyondmimic_walk_contact.yaml` | 步态更流畅，脚步稳定性较低 |

这里的7996来自 `TASK_20261103_048`，与此前被否定的同名检查点不是同一权重。模型来源、SHA和配置绑定见 [walk_models.json](deploy/robots/g1_29dof/config/policy/beyondmimic/walk_models.json)。

两版均使用154维 actor，只接收参考、IMU、编码器和上一动作，不需要位置或线速度估计；原生训练已执行相同的关节、目标速度及 PD 包络限制。前推控制行走速度，行走时右杆控制转向，回中按支撑相制动并保持策略站立，再推杆重新起步。侧移、倒退和原地转向不在训练范围内。

模型完整性、Linux编译、推理一致性及离线闭环结果见 [验证记录](docs/beyondmimic_walk_validation.json)。5997的原冻结功能协议全部通过；7996仍有初始站立和停车速度/偏航未过门项。两版按用户要求保留比较，尚无真机验收，不能把打包和编译通过当成整体行为通过。

更新和构建不会启动机器人：

```bash
cd ~/G1_deploy
git switch yanni-G1-walk-beyondmimic
git pull --ff-only origin yanni-G1-walk-beyondmimic
python3 tools/check_beyondmimic_walk_package.py
bash tools/build_g1_deploy.sh
./deploy/robots/g1_29dof/build/g1_ctrl --help
```

- [从拉取、构建到人工选择模型](docs/G1_真机从git_pull到运行.md)
- [手柄操作、模型接口与实际验收状态](docs/G1_BeyondMimic_行走模型.md)
- [154维接口及历史修正记录](docs/beyondmimic_proprio154.md)
- [旧 dance_zui 有限片段说明](docs/G1_BeyondMimic_部署流程.md)，使用独立配置 `beyondmimic.yaml`。

真机推理无需 Isaac Lab、PyTorch 或训练仓库。控制框架来自 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)，训练架构基于 [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)。原始许可见 [LICENCE](LICENCE)。

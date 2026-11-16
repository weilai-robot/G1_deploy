# guohua 首帧高 Kp 保持测试

该配置用于确认首帧的静态支撑情况，**不启动策略、不播放舞蹈**。正常 `guohua_show_dance` 的 48 秒配置保留。

之前的首帧确认发生在降低到训练 PD 之后。诊断配置改为从原 FixStand 过渡到首帧，所有腿部关节 Kp 使用 150，腰部使用 200，手臂使用 40；Kd 全部沿用原站立值，踝关节为 3。目标姿态和训练模型没有改变。高 Kp 不代表首帧可以保持平衡，需使用防跌落支撑。

## 更新与启动

在机载 Linux 工程执行：

```bash
cd ~/G1_deploy
git pull --ff-only origin main
bash tools/build_g1_deploy.sh

cd deploy/robots/g1_29dof
./build/g1_ctrl --network eth0 --domain 0 \
  --config config/beyondmimic_guohua_first_frame.yaml --log
```

`eth0` 替换为实际机器人网卡。诊断配置使用独立的 `BeyondMimicFirstFrameTest` 状态类型，旧二进制会报未知类型并拒绝启动，避免忽略新选项后误运行舞蹈；不要将该类型改回 `BeyondMimic` 来绕过编译更新。启动控制器前按宇树流程进入开发模式，保持防跌落支撑。确认日志显示：

```text
FIRST FRAME TEST ONLY: policy inference DISABLED
First-frame PD: hip_pitch L/R=150/150 knee L/R=150/150 ankle_pitch L/R=150/150
```

1. L2+上：进入原 FixStand 站姿，踝关节仍为 150/3。
2. 等站稳后，L2 持续至少 2 秒+右：快速过渡到 guohua 首帧。
3. 首帧确认通过后显示 `FIRST FRAME HOLD reached`，保持高 PD，**不会切入舞蹈推理**。
4. L2+B：退出到阻尼。退出前确保吊装可以承重；Ctrl+C 也会请求阻尼退出。

测试模式没有正常舞蹈的 10 秒或 48 秒计时停止。它保持到手动退出或保护触发，应短时观察并由操作员监护；不要通过触摸通电关节判断支撑力。

## 日志与退出

保持阶段每秒记录最大关节误差、速度、骨盆倾斜，以及双膝/双踝 pitch 的 `q/target/tau_PD`。角度单位为 rad，`tau_PD = Kp × (target-q) - Kd × dq` 是根据当前命令和反馈计算的估计值，**不是电机实际反馈力矩**。

关节限位、目标变化速度、PD 估算力矩约束、通信/输出超时和骨盆倾斜检查保持启用。进入保持后若丢失姿态/速度到位条件则退出到阻尼，不会在失败时改为运行策略。

比较 `q` 与 `target`：持续明显跟不上目标，需要进一步检查支撑需求、执行器反馈和标定；角度已到位仍倾倒，则不能仅靠继续加 Kp 判断或解决平衡问题。

诊断参数 `first_frame_kp`、`first_frame_kd` 是 SDK 电机顺序的 29 元素数组，仅在 `first_frame_test: true` 时使用，不修改正常舞蹈的训练 PD。需要重新编译本次 C++ 修改；配置和无硬件规划器测试通过不等于真机稳定性验证。

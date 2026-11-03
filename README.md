# G1 29DOF：11k 基线真机测试

**终端命令在 G1 机载 Linux 电脑执行；机器人动作通过宇树遥控手柄控制。** 当前使用训练仓库 `unitree_rl_lab_train` 的 `model_10999_factorygait.pt`，对应 ONNX 和参数已包含在本仓库。

已完成 Windows C++ 推理对照及 MuJoCo 闭环检查；**尚未在 Linux 编译 DDS 主控制器，尚未做真机测试**。先完成编译、自检及[部署控制器仿真检查](docs/G1_11k_部署流程.md#先用部署控制器做-dds-仿真)，再进行受保护的真机试验。首次测试保持防跌落保护，退出策略会进入阻尼，不能继续保持站立。

## 1. 下载代码

首次下载：

```bash
cd ~
git clone --branch yanni-G1-walk-01 https://github.com/weilai-robot/G1_deploy.git G1_deploy
cd ~/G1_deploy
```

已经有仓库则执行以下命令；有本地修改时先保留：

```bash
cd ~/G1_deploy
git status --short
git fetch origin
git switch yanni-G1-walk-01
git pull --ff-only origin yanni-G1-walk-01
```

## 2. 编译

首次使用 Ubuntu 时安装依赖；已安装可跳过：

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl python3 \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

校验模型并编译：

```bash
cd ~/G1_deploy
python3 tools/check_velocity11k_package.py
bash tools/build_g1_deploy.sh
```

模型校验应输出 `PASS`，编译应生成 `deploy/robots/g1_29dof/build/g1_ctrl`。构建脚本还会自动运行离线推理对照检查，输出 `"result": "PASS"`；该检查不会发送机器人控制指令。检查失败时先排查，不进入启动步骤。

如果 SDK 安装在 `/opt/unitree_robotics`，把最后一行换成：

```bash
UNITREE_SDK_PREFIX=/opt/unitree_robotics bash tools/build_g1_deploy.sh
```

若提示缺少 `unitree/dds_wrapper/robots/g1/g1.h`，先按 [SDK 安装说明的第 3 节](docs/G1_BeyondMimic_部署流程.md#3-安装-linux-编译依赖与宇树-sdk)配置包含该头文件的 SDK，再重新编译。

## 3. 启动

先按本机固件操作流程进入开发/调试状态，停止内置运动控制及其他低层控制程序。

查看网卡：

```bash
ip -br -4 addr
```

启动程序。下面的 `eth0` **必须替换成实际连接机器人网络的网卡**：

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
./build/g1_ctrl --network eth0 --domain 0 \
  --config config/config.yaml --log
```

看到 `Connected to robot.`、`Velocity11k loaded: checkpoint=...` 和 `FSM: Start Passive` 后，保持终端运行。**这时尚未开始行走。**

## 4. 用手柄控制

以下按键与当前 [11k 配置](deploy/robots/g1_29dof/config/config.yaml)的 `transitions`、`FixStand.ts` 一致；代码中的 `LT` 对应手柄 **L2**，`RB` 对应 **R1**。组合键先按住肩键，再按另一按键触发。

| 顺序 | 宇树遥控手柄操作 | 机器人行为 |
|---|---|---|
| ① | **L2 + 方向键上** | 进入准备站姿；至少等待配置中的 **3 秒**，确认姿态和双脚接触正常，两个摇杆回中 |
| ② | **R1 + X** | 进入 11k 行走策略，随后通过摇杆给定速度 |
| ③ | 左摇杆前 / 后 | 前进 / 后退 |
| ④ | 左摇杆左 / 右 | 向左 / 向右侧移 |
| ⑤ | 右摇杆左 / 右 | 左转 / 右转 |
| ⑥ | **两个摇杆回中** | 给定零速度命令，策略继续运行，用于正常停车 |
| ⑦ | **L2 + B** | 退出策略，进入阻尼，由防跌落保护承重 |
| ⑧ | 在终端按 **Ctrl+C** | 请求阻尼并结束程序 |

摇杆速度命令随 SDK 返回的轴值连续变化。策略会持续运行，直到手动退出或触发故障保护。**正常停车先将摇杆回中；L2 + B 会进入阻尼，不能保持站立。** 退出后重新启动，需再次执行 `L2 + 方向键上`，等待站姿过渡完成，再按 `R1 + X`。

日志在 `~/G1_deploy/deploy/robots/g1_29dof/log/`，测试后保留该目录及视频。模型接口、速度范围和检查边界见[详细说明](docs/G1_11k_部署流程.md)，实际验证结果见[验证记录](docs/velocity11k_validation.json)。

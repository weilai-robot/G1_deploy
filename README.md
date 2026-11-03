# G1 29DOF：BeyondMimic 真机测试

**终端命令在 G1 机载 Linux 电脑执行；机器人动作通过宇树遥控手柄控制。** 下面先测试 5997 稳定版。

当前尚未完成真机验收；7996 流畅版仍有初始站立、停车未通过的离线检查项。首次测试保持防跌落保护，退出策略会进入阻尼，不能继续保持站立。

## 1. 下载代码

首次下载：

```bash
cd ~
git clone --branch yanni-G1-walk-beyondmimic https://github.com/weilai-robot/G1_deploy.git G1_deploy
cd ~/G1_deploy
```

已经有仓库则执行以下命令；有本地修改时先保留：

```bash
cd ~/G1_deploy
git fetch origin
git switch yanni-G1-walk-beyondmimic
git pull --ff-only origin yanni-G1-walk-beyondmimic
```

## 2. 编译

首次使用 Ubuntu 时安装依赖；已安装可跳过：

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl python3 \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

校验模型并编译（脚本会按本机架构下载 ONNX Runtime）：

```bash
cd ~/G1_deploy
python3 tools/check_beyondmimic_walk_package.py
bash tools/build_g1_deploy.sh
```

模型校验应输出 `"selected_pair": "passed"`，编译应生成 `deploy/robots/g1_29dof/build/g1_ctrl`。

如果 SDK 安装在 `/opt/unitree_robotics`，把最后一行换成：

```bash
UNITREE_SDK_PREFIX=/opt/unitree_robotics bash tools/build_g1_deploy.sh
```

若提示缺少 `unitree/dds_wrapper/robots/g1/g1.h`，按[SDK 安装说明](docs/G1_BeyondMimic_部署流程.md#3-安装-linux-编译依赖与宇树-sdk)配置包含该头文件的 SDK，再重新编译。

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
  --config config/beyondmimic_walk.yaml --log
```

看到 `Connected to robot.`、模型标识 `stable_5997_r26` 和 `FSM: Start Passive` 后，保持终端运行。**这时尚未开始走路。**

| 模型 | 启动配置（--config） | 对比观感 |
|---|---|---|
| 5997 稳定版（先测） | `config/beyondmimic_walk.yaml` | 脚步更稳，流畅度较低 |
| 7996 流畅版 | `config/beyondmimic_walk_contact.yaml` | 步态更流畅，脚步稳定性较低 |

测试 7996 时，**先退出当前控制程序**，再将启动配置换成表中对应项；日志模型标识应为 `fluid_7996_r29`。两版共用同一次编译结果。

## 4. 用手柄控制

以下按键与当前[行走配置](deploy/robots/g1_29dof/config/beyondmimic_walk.yaml)一致；准备姿态过渡时间来自 `FixStand.ts`：

| 顺序 | 宇树遥控手柄操作 | 机器人行为 |
|---|---|---|
| ① | **L2 + 方向键上** | 进入当前模型准备站姿；至少等待配置中的 **3 秒**，确认姿态和双脚接触正常 |
| ② | **左右摇杆回中，再按 R1 + X** | 进入行走策略，先保持策略站立 |
| ③ | **左摇杆前推** | 开始行走；前推幅度控制前进速度 |
| ④ | **前进时拨动右摇杆左右方向** | 控制行走转向 |
| ⑤ | **左右摇杆回中** | 按支撑相制动并保持策略站立；再次前推可重新起步 |
| ⑥ | **L2 + B** | 退出策略，进入阻尼，由防跌落保护承重 |
| ⑦ | 在终端按 **Ctrl+C** | 请求阻尼并结束程序 |

**不支持倒退、侧移或原地转向。摇杆回中用于策略停车；L2 + B 是退出到阻尼，不是“平稳停下并站住”。**

日志在 `~/G1_deploy/deploy/robots/g1_29dof/log/beyondmimic_walk/`，文件名包含模型标识。测试后保留完整 CSV 和视频。模型接口、验证结果与旧模型保留情况见[详细说明](docs/G1_BeyondMimic_行走模型.md)。

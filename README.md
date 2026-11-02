# G1 29DOF：HugWBC 真机测试

**终端命令在 G1 机载 Linux 电脑执行；机器人动作通过宇树遥控手柄控制。** 下面先测试 R41。

本次复审修正目前仍在本地，尚未提交推送；先同步完整修正版，再按下面步骤下载。首次测试保持防跌落保护，退出策略会进入阻尼，不能继续保持站立。

## 1. 下载代码

首次下载：

```bash
cd ~
git clone --branch yanni-G1-walk-hugWBC https://github.com/weilai-robot/G1_deploy.git G1_deploy
cd ~/G1_deploy
```

已经有仓库则执行以下命令；有本地修改时先保留：

```bash
cd ~/G1_deploy
git fetch origin
git switch yanni-G1-walk-hugWBC
git pull --ff-only origin yanni-G1-walk-hugWBC
```

## 2. 编译

首次使用 Ubuntu 时安装依赖；已安装可跳过：

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl python3 python3-yaml \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

校验模型并编译：

```bash
cd ~/G1_deploy
python3 tools/check_hugwbc_package.py
bash tools/build_g1_hugwbc.sh r41
```

模型校验应输出 `PASS`，编译应生成 `deploy/robots/g1_29dof/build/g1_ctrl`。

如果 SDK 安装在 `/opt/unitree_robotics`，把最后一行换成：

```bash
UNITREE_SDK_PREFIX=/opt/unitree_robotics bash tools/build_g1_hugwbc.sh r41
```

若提示缺少 `unitree/dds_wrapper/robots/g1/g1.h`，先按[宇树 SDK 安装说明](https://github.com/unitreerobotics/unitree_sdk2/blob/main/README.md)配置 SDK，再重新编译。

## 3. 启动

先按本机固件操作流程进入开发/调试状态，停止内置运动控制及其他低层控制程序。

查看网卡：

```bash
ip -br -4 addr
```

启动程序。下面的 `eth0` 必须替换成实际连接机器人网络的网卡：

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
./build/g1_ctrl --network eth0 --domain 0 \
  --config config/hugwbc_r41.yaml --log
```

看到 `Connected to robot.`、`HugWBC loaded: package=r41 ...` 和 `FSM: Start Passive` 后，保持终端运行。**这时尚未开始走路。**

测试 R42 时，将编译命令末尾的 `r41` 改为 `r42`，启动配置改为 `config/hugwbc_r42.yaml`。

## 4. 用手柄控制

以下按键与当前 [HugWBC 配置](deploy/robots/g1_29dof/config/hugwbc_r41.yaml)的 `transitions`、`FixStand.ts` 一致：

| 顺序 | 宇树遥控手柄操作 | 机器人行为 |
|---|---|---|
| ① | **L2 + 方向键上** | 进入准备站姿；至少等待配置中的 **3 秒**，确认姿态和双脚接触正常 |
| ② | **持续按住 L2 至少 2 秒，再按方向键右** | 开始 HugWBC 固定前行 |
| ③ | **L2 + B** | 退出行走，进入阻尼，由防跌落保护承重 |
| ④ | 在终端按 **Ctrl+C** | 请求阻尼并结束程序 |

**当前靠手柄按键切换状态，摇杆不能调速或转向，也不使用 R1 + X 启动。** 行走会持续，直到主动退出或触发故障保护；L2 + B 不是“平稳停下并站住”。

日志在 `~/G1_deploy/deploy/robots/g1_29dof/log/`，测试后保留该目录及视频。模型接口与验证记录见[详细说明](docs/G1_HugWBC_部署流程.md)。

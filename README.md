# G1_deploy：G1 EDU U2 真机操作步骤

本仓库基于宇树官方 `unitree_rl_lab`，用于在 **G1 EDU U2（29DOF）** 上部署 BeyondMimic `dance_zui` 模型。模型和参考动作已包含在仓库中。

**当前为默认 10 秒的吊装试验版本：已完成仿真和编译检查，尚未进行真机测试，整段动作未通过部署验证。试验结束会进入阻尼，需要吊装承重。**

先通过现有 SSH 或远程终端登录 **G1 机载开发电脑**。以下终端命令均在 G1 上执行，项目目录以 `~/G1_deploy` 为例。

## 1. 拉取部署代码

已经克隆过仓库：

```bash
cd ~/G1_deploy
git switch main
git pull --ff-only origin main
```

首次下载仓库：

```bash
cd ~
git clone https://github.com/weilai-robot/G1_deploy.git G1_deploy
cd ~/G1_deploy
```

如果之前克隆的是宇树原仓库，先核对 `git remote -v`。本项目的 `origin` 应指向 `weilai-robot/G1_deploy`；具体修改方式见[详细操作文档](docs/G1_真机从git_pull到运行.md)。

## 2. 检查系统和模型

```bash
cd ~/G1_deploy
uname -m
cat /etc/os-release
python3 tools/check_beyondmimic_package.py
```

模型检查应输出：

```text
PASS: policy and reference checksums, 160 observations / 29 actions, 50 Hz, 11384 frames
```

`aarch64`/`arm64` 表示 ARM64，`x86_64` 表示 Intel/AMD。构建脚本会选择对应的运行库。真机运行不需要安装 Isaac Lab、训练仓库、PyTorch 或 ROS。

## 3. 首次安装编译依赖和宇树 SDK

已安装所需依赖与 SDK 时，跳到第 4 步。首次配置 Ubuntu 机载开发电脑时执行：

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl python3 \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

部署需要的 SDK 头文件为 `unitree/dds_wrapper/robots/g1/g1.h`。如果它已位于 `/usr/local/include` 或 `/opt/unitree_robotics/include`，可跳过下面的 SDK 安装。

未安装所需 SDK 时：

```bash
mkdir -p ~/g1_dependencies
cd ~/g1_dependencies

if [ ! -d unitree_sdk2/.git ]; then
  git clone https://github.com/unitreerobotics/unitree_sdk2.git
fi

cmake -S unitree_sdk2 -B unitree_sdk2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build unitree_sdk2/build -j2
sudo cmake --install unitree_sdk2/build
sudo ldconfig
```

## 4. 编译部署控制器

SDK 安装在默认的 `/usr/local` 时：

```bash
cd ~/G1_deploy
bash tools/build_g1_deploy.sh
```

如果 SDK 安装在 `/opt/unitree_robotics`，改用：

```bash
cd ~/G1_deploy
UNITREE_SDK_PREFIX=/opt/unitree_robotics bash tools/build_g1_deploy.sh
```

ARM64 首次构建需要联网，从微软官方 GitHub Release 下载 ONNX Runtime 1.22.0。已有对应运行库时会直接使用。

编译完成后，检查程序和动态库：

```bash
cd ~/G1_deploy
./deploy/robots/g1_29dof/build/g1_ctrl --help
ldd ./deploy/robots/g1_29dof/build/g1_ctrl
```

`--help` 应正常显示参数，`ldd` 中不应出现 `not found`。以上检查和编译不会启动机器人控制。

## 5. 准备真机并确认网卡

保持吊装防跌落保护，按本机固件手册从零力矩状态进入开发/调试模式，常见按键为 **L2 + R2**。确认内置运动控制程序已停止发送控制指令。

在 G1 终端查看网卡：

```bash
ip -br -4 addr
```

选择机器人内部网络所在的网卡，常见是带有 `192.168.123.x` 地址的接口。下面以 `eth0` 为例，按实际网卡名称修改：

```bash
G1_NET=eth0
ip -4 addr show "$G1_NET"
```

真机使用 DDS domain `0`。

## 6. 启动控制器

在设置了 `G1_NET` 的同一个终端执行：

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
./build/g1_ctrl --network "$G1_NET" --domain 0 \
  --config config/beyondmimic.yaml --log
```

正常连接后应看到：

```text
Connected to robot.
BeyondMimic loaded: 160 obs / 29 actions, ...
FSM: Start Passive
```

保持这个终端运行。此时为阻尼状态，尚未启动策略。

## 7. 遥控器操作与退出

| 顺序 | 操作 | 结果 |
|---|---|---|
| 1 | 按 **L2 + 方向键上** | 进入 FixStand，用 3 秒过渡到训练默认站姿 |
| 2 | 等待至少 3 秒，让双脚稳定接触平整地面，保持防跌落保护 | 为腿部里程计提供接触条件 |
| 3 | 持续按住 **L2 至少 2 秒**，再按 **方向键右** | 启动 BeyondMimic，运行默认 10 秒策略段 |
| 4 | 按 **L2 + B** | 退出策略，进入阻尼 |
| 5 | 默认 10 秒结束 | 自动进入阻尼，需要吊装承重 |

双脚悬空时不启动本策略。当前配置通过参考动作控制，不使用 `R1 + X` 启动，也不通过摇杆改变动作。

正常结束进程时，在控制器终端按 **Ctrl+C**；程序会请求发布阻尼后退出。阻尼状态不会继续保持站立，不用关闭 SSH 窗口代替正常退出。

## 8. 以后每次更新

依赖与 SDK 已配置好后，每次更新通常只需：

```bash
cd ~/G1_deploy
git switch main
git pull --ff-only origin main
python3 tools/check_beyondmimic_package.py
bash tools/build_g1_deploy.sh
```

SDK 在 `/opt/unitree_robotics` 时，仍使用第 4 步带 `UNITREE_SDK_PREFIX` 的构建命令。编译后按第 5～7 步准备和启动真机。

查看最近日志：

```bash
tail -n 100 ~/G1_deploy/deploy/robots/g1_29dof/log/log.txt
```

## 进一步说明

- [逐步操作、预期输出与故障排查](docs/G1_真机从git_pull到运行.md)
- [模型接口、仿真验证与已知限制](docs/G1_BeyondMimic_部署流程.md)
- [实际验证记录](docs/beyondmimic_validation.json)

本项目保留宇树官方 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) 的控制框架，模型训练基于 [BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)。原始项目许可见 [LICENCE](LICENCE)。

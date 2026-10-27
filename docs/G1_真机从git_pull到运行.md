# G1 EDU U2：远程登录后，从 git pull 到真机试运行

适用对象：G1 EDU U2、29DOF，运行本仓库的 BeyondMimic `dance_zui` 模型。

**当前默认只执行最初 10 秒，结束后进入阻尼，需要吊装承重。10 秒闭环仿真已通过，尚未进行真机测试，整段动作未通过部署验证。** 本文的启动操作用于吊装试验。

## 第 1 步：远程登录 G1 的开发电脑

用你现有的 SSH 或远程终端连接 G1 的机载开发电脑。**下文所有终端命令都在 G1 上执行**，不在本地 Mac 执行。不要登录或修改宇树专用运动控制计算单元。

本文假设部署仓库在 `~/G1_deploy`。如果你放在其他目录，只需要替换相应的 `cd` 路径。

## 第 2 步：拉取部署代码和模型

已经克隆过仓库时：

```bash
cd ~/G1_deploy
git remote -v
git status --short
git switch main
git pull --ff-only origin main
git log -1 --oneline
```

`origin` 应指向 `weilai-robot/G1_deploy`。如果这里仍指向宇树的 `unitree_rl_lab`，先修改，再重新执行上面的 pull：

```bash
git remote set-url origin https://github.com/weilai-robot/G1_deploy.git
git pull --ff-only origin main
```

如果 G1 上还没有这个项目，只在首次执行：

```bash
cd ~
git clone https://github.com/weilai-robot/G1_deploy.git G1_deploy
cd ~/G1_deploy
```

如果 Git 提示本地修改冲突或不能快进，先保留并处理本地修改，再继续；本文不需要执行强制重置。

## 第 3 步：检查系统和模型文件

```bash
cd ~/G1_deploy
uname -m
cat /etc/os-release
python3 tools/check_beyondmimic_package.py
```

预期看到：

```text
PASS: policy and reference checksums, 160 observations / 29 actions, 50 Hz, 11384 frames
```

`PASS` 表示模型和参考动作文件完整。模型已在仓库中，不需要在 G1 重新导出，也不需要安装 Isaac Lab、训练仓库、PyTorch 或 ROS。

`uname -m` 输出 `aarch64`/`arm64` 时，需要 ARM64 运行库；输出 `x86_64` 时，需要 x86_64 运行库。后面的构建脚本会自动选择。

如果 `tools/check_beyondmimic_package.py` 不存在，说明拉到的代码不包含本次适配。先核对远端地址和最新提交，不进入后续启动步骤。

## 第 4 步：首次安装编译依赖

以下是 Ubuntu 机载开发电脑上的安装步骤。先确认第 3 步的系统信息；这一步只在首次配置或依赖缺失时执行。

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl python3 \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

已配置好的电脑以后更新代码时，可以跳过这一步。

## 第 5 步：检查并安装宇树 SDK

先检查部署代码需要的 SDK 头文件：

```bash
if [ -f /usr/local/include/unitree/dds_wrapper/robots/g1/g1.h ]; then
  echo "SDK 已安装在 /usr/local，跳过安装步骤"
elif [ -f /opt/unitree_robotics/include/unitree/dds_wrapper/robots/g1/g1.h ]; then
  echo "SDK 已安装在 /opt/unitree_robotics，编译时使用该路径"
else
  echo "未发现所需 SDK，请执行下面的安装步骤"
fi
```

如果已找到所需 SDK，直接进入第 6 步。如果没有，执行：

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

若这个依赖目录之前已存在，但构建仍提示缺少 `dds_wrapper`，需确认该目录的 SDK 版本包含上述头文件。

## 第 6 步：编译部署控制器

SDK 在默认的 `/usr/local` 时：

```bash
cd ~/G1_deploy
bash tools/build_g1_deploy.sh
```

SDK 在 `/opt/unitree_robotics` 时，改用：

```bash
cd ~/G1_deploy
UNITREE_SDK_PREFIX=/opt/unitree_robotics bash tools/build_g1_deploy.sh
```

构建脚本会检查模型、选择 CPU 架构并编译。ARM64 首次构建需要联网，从微软官方 GitHub Release 下载 ONNX Runtime 1.22.0；后续已有运行库时不会重复下载。

成功后应存在：

```text
deploy/robots/g1_29dof/build/g1_ctrl
deploy/robots/g1_29dof/build/beyondmimic_offline
```

先做不会发送机器人控制指令的检查：

```bash
cd ~/G1_deploy
./deploy/robots/g1_29dof/build/g1_ctrl --help
ldd ./deploy/robots/g1_29dof/build/g1_ctrl
```

`--help` 应正常显示参数；`ldd` 中不应出现 `not found`。**编译结束不会自动控制机器人。**

## 第 7 步：准备真机并选网卡

保持吊装防跌落保护。按本机固件手册，从零力矩状态进入开发/调试模式；常见按键为 `L2 + R2`。确认内置运动控制程序已停止发送控制指令。这些机身与遥控器操作由现场完成。

在 G1 终端查看网卡：

```bash
ip -br -4 addr
```

找到机器人内部网络所在的网卡，常见是带有 `192.168.123.x` 地址的接口。以实际结果为准；机载电脑的接口名称不一定就是 `eth0`。

下面假设选中的是 `eth0`：

```bash
G1_NET=eth0
ip -4 addr show "$G1_NET"
```

如果实际是其他网卡，修改 `G1_NET`。真机使用 DDS domain `0`；`lo` 与 domain `1` 是本文仿真流程的配置。

## 第 8 步：启动控制程序

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
./build/g1_ctrl --network "$G1_NET" --domain 0 \
  --config config/beyondmimic.yaml --log
```

保持这个终端运行。正常连接后，应依次看到相关提示：

```text
Connected to robot.
BeyondMimic loaded: 160 obs / 29 actions, ...
FSM: Start Passive
```

此时是阻尼状态，尚未运行策略。

## 第 9 步：现场按遥控器开始试验

| 操作 | 预期结果 |
|---|---|
| 按 `L2 + 方向键上` | 进入 `FixStand`，用 3 秒过渡到训练默认站姿 |
| 等待至少 3 秒，保持防跌落保护，让双脚稳定接触平整地面 | 给腿部里程计提供有效的接触条件 |
| 持续按住 `L2` 至少 2 秒，再按 `方向键右` | 进入 `BeyondMimic`，开始默认 10 秒策略段 |
| 按 `L2 + B` | 退出策略，进入 `Passive` 阻尼 |
| 默认 10 秒结束 | 自动进入阻尼，需要吊装承重 |

不要在双脚悬空时开始本策略，腿部里程计需要支撑脚接地。此配置不使用 `R1 + X` 启动，也不通过摇杆改变参考动作。

正常结束进程时，在运行控制器的终端按 `Ctrl+C`。程序会请求发布阻尼后退出。默认试验结束和阻尼退出都不会继续保持站立。

## 第 10 步：查看日志与排查

可以另开一个 G1 终端查看：

```bash
tail -n 100 ~/G1_deploy/deploy/robots/g1_29dof/log/log.txt
```

| 提示或现象 | 下一步 |
|---|---|
| 模型校验失败 | 检查代码更新和模型文件是否完整，重新拉取对应版本 |
| 首次构建下载 ONNX Runtime 失败 | 检查 G1 是否能访问 GitHub；也可在联网电脑下载匹配架构的运行库，再传入对应 `deploy/thirdparty` 目录 |
| `unitree/dds_wrapper/robots/g1/g1.h` 找不到 | 检查第 5 步的 SDK 版本和安装路径 |
| 动态库 `not found` / 程序加载库失败 | 运行 `sudo ldconfig` 并重新检查 `ldd`；确认运行库架构正确 |
| 一直停在 `Waiting for connection to robot` | 核对网卡、内部网络、机器人是否启动，以及 DDS domain 是否为 0 |
| `The other process is using the lowcmd channel` | 确认原控制程序或其他部署进程已退出，再重新启动 |
| `Initial posture differs from reference` | 保持保护，重新进入 `FixStand`，等待站姿稳定后再切换 |
| 策略刚进入就回到 `Passive` | 查看紧邻的 `BeyondMimic -> Passive` 原因，核对超时、IMU、姿态及估计误差；不要反复直接重启策略 |
| `Segment completed` | 正常的默认 10 秒结束，并非程序故障 |

## 以后每次 git pull 更新后怎么做

首次依赖与 SDK 已安装后，通常只需要：

```bash
cd ~/G1_deploy
git switch main
git pull --ff-only origin main
python3 tools/check_beyondmimic_package.py
bash tools/build_g1_deploy.sh
```

随后按第 7～9 步进行真机准备和启动。如果 SDK 安装在 `/opt/unitree_robotics`，上面的构建命令仍需加上对应的 `UNITREE_SDK_PREFIX`。

完整的模型接口、仿真验证、已知限制与重新导出方法见 [G1_BeyondMimic_部署流程.md](G1_BeyondMimic_部署流程.md)，实际验证记录见 [beyondmimic_validation.json](beyondmimic_validation.json)。

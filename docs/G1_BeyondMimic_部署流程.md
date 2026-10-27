# G1 EDU U2：BeyondMimic 模型部署流程

## 当前状态与使用范围

本项目已经接入 `Beyondmimic_Weilai_G1` 中的 `dance_zui_model_4999.pt` 和对应的 `dance_zui.npz`，使用宇树 `unitree_rl_lab` 的 G1 29DOF 控制程序、DDS 通信和状态机。

如果你已经远程登录 G1，准备通过 `git pull` 获取部署代码，请先看 [远程登录后，从 git pull 到真机试运行](G1_真机从git_pull到运行.md)。该文档按操作顺序区分首次安装和日常更新。

**当前是受保护试验版本，不是已验证的整段真机动作。** 默认从第 0 帧运行 10 秒，随后进入阻尼。10 秒闭环仿真通过；更长动作在部署限制下没有通过。保持吊装防跌落保护，阻尼会失去站立支撑。

已有的训练检查点本身不会因部署而改变。部署包中的 ONNX 已包含经验归一化，C++ 端直接输入原始 160 维观测，不能再做一次归一化。

| 内容 | 已确认的配置 |
|---|---|
| 机器人 | G1 EDU U2，29 个身体关节，不含手部独立控制 |
| 模型 | 160 → 512 → 256 → 128 → 29，ELU |
| 策略周期 | 20 ms / 50 Hz |
| 状态机、发送周期 | 1 ms / 1000 Hz；模型推理使用独立线程 |
| 动作 | `default_joint_pos + action_scale × raw_action`，随后执行部署限幅 |
| 观测 | 参考关节位置/速度、相对躯干位置/朝向、机体线/角速度、关节位置偏差/速度、上次原始动作 |
| 参考动作 | 11384 帧，50 Hz；默认只选择最初 500 帧 |
| 状态估计 | 关节编码器 + 骨盆 IMU + 双腿运动学里程计 |

## 1. 代码与模型位置

以下路径均相对于项目根目录：

```text
deploy/robots/g1_29dof/config/beyondmimic.yaml       专用启动配置
deploy/robots/g1_29dof/include/State_BeyondMimic.h
deploy/robots/g1_29dof/src/State_BeyondMimic.cpp     独立策略状态
deploy/include/beyondmimic/core.h                  观测、FK、里程计、目标限幅
deploy/include/beyondmimic/package.h               包加载与 ONNX 推理
deploy/robots/g1_29dof/config/policy/beyondmimic/dance_zui/
  policy.onnx                                    已转换的推理模型
  reference.bin                                  已转换的参考动作
  deploy.yaml                                    训练参数、关节映射、运动学
  manifest.json                                  原始模型、动作和产物校验值
tools/export_beyondmimic.py                       独立导出，无需 Isaac Lab
tools/check_beyondmimic_package.py                无第三方依赖的完整性检查
tools/verify_beyondmimic.py                       使用实际 C++ 核心做 MuJoCo 验证
tools/build_g1_deploy.sh                          Linux 编译与架构选择
```

原来的 `config/config.yaml` 和官方预训练策略仍可单独使用。本次接入需要明确传入 `--config`。

## 2. 在机载开发电脑准备项目

在机载开发电脑准备包含此次新增代码和模型的项目目录，例如 `~/G1_deploy`。通过 Git 获取时使用 `https://github.com/weilai-robot/G1_deploy.git`，已有本仓库时在项目目录执行 `git pull --ff-only origin main`。也可以使用交付压缩包，解压后即为 `G1_deploy`。应使用机载开发计算单元，不改动宇树运动控制计算单元的系统文件。

在机载开发电脑执行：

```bash
cd ~/G1_deploy
uname -m
python3 tools/check_beyondmimic_package.py
```

最后一条应输出 `PASS`。`aarch64` 是 ARM64；`x86_64` 是 Intel/AMD。不要将仓库自带的 Linux x86_64 ONNX Runtime 库用于 ARM64。

## 3. 安装 Linux 编译依赖与宇树 SDK

以下适用于 Ubuntu 22.04/24.04。已经正确安装 SDK 时，可以跳过 SDK 下载和安装步骤。

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl python3 \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

mkdir -p ~/g1_dependencies
cd ~/g1_dependencies
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cmake -S unitree_sdk2 -B unitree_sdk2/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build unitree_sdk2/build -j2
sudo cmake --install unitree_sdk2/build
sudo ldconfig

cd ~/G1_deploy
bash tools/build_g1_deploy.sh
```

构建脚本会根据当前电脑架构选择 ONNX Runtime 1.22.0。ARM64 首次构建时会从微软官方 Release 下载对应运行库。成功后生成：

```text
deploy/robots/g1_29dof/build/g1_ctrl
deploy/robots/g1_29dof/build/beyondmimic_offline
```

如果 SDK 安装在 `/opt/unitree_robotics`，改用：

```bash
UNITREE_SDK_PREFIX=/opt/unitree_robotics bash tools/build_g1_deploy.sh
```

若提示缺少 `unitree/dds_wrapper/robots/g1/g1.h`，当前 SDK 版本不包含官方部署框架需要的封装，需安装包含该头文件的 SDK 版本。

## 4. 先验证当前模型与部署链路

仿真验证可以在独立 Linux 电脑进行。真实控制器运行不需要 PyTorch、Isaac Lab 或训练仓库。

### 4.1 使用训练机器人的 MJCF 做无硬件闭环验证

准备训练仓库中的 `assets/unitree_description` 完整目录，保留 MJCF 与 meshes 的相对路径。安装验证依赖：

```bash
python3 -m venv .venv-verify
source .venv-verify/bin/activate
pip install numpy onnxruntime mujoco scipy PyYAML

python tools/verify_beyondmimic.py \
  --offline-binary deploy/robots/g1_29dof/build/beyondmimic_offline \
  --mjcf /实际训练仓库/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml \
  --seconds 10 --report /tmp/g1_beyondmimic_verification.json
```

默认使用腿部里程计。程序会检查文件校验值、64 组 C++ 观测和动作对照、异常 IMU 拒绝以及闭环稳定性。`closed_loop_survived` 应为 `true`；失败时程序返回非零退出码。

`--state-source truth` 只供仿真诊断，会向离线程序提供仿真位置/速度；真机状态不使用该入口。`--raw-targets` 也仅供仿真诊断，不代表可在真机取消部署限幅。

### 4.2 使用宇树官方 MuJoCo 验证 DDS 与遥控器流程

安装官方 `unitree_mujoco`，选择 G1 的 29DOF 场景，使用游戏手柄并启用吊装弹性带。仿真与控制器使用相同的 DDS domain，例如 1，以及回环网卡 `lo`。

仿真进程（路径按实际安装调整）：

```bash
cd ~/g1_dependencies/unitree_mujoco/simulate/build
./unitree_mujoco -i 1 -n lo -r g1 -s scene_29dof.xml
```

控制进程：

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
./build/g1_ctrl --network lo --domain 1 --config config/beyondmimic.yaml --log
```

按下节的站姿和策略切换流程操作手柄。仿真域与真机域保持分离。官方模拟器模型与训练 MJCF 可能不同，需要单独验证。

## 5. 吊装真机启动流程

1. 保持吊装防跌落保护，在零力矩状态启动 G1。按本机固件手册进入开发/调试模式；常见流程是在零力矩模式按 `L2 + R2`。确认内置运动控制程序已停止发送控制指令。
2. 在机载开发电脑用 `ip -br addr` 查找机器人内部网络对应的网卡，例如 `eth0`。如果在外部 Linux 电脑运行，使用连接机器人的有线网卡。
3. 启动下面的控制程序。把 `eth0` 替换成实际网卡；真机 DDS domain 为 0。

```bash
cd ~/G1_deploy/deploy/robots/g1_29dof
./build/g1_ctrl --network eth0 --domain 0 --config config/beyondmimic.yaml --log
```

4. 等待 `Connected to robot`、`BeyondMimic loaded: 160 obs / 29 actions` 和 `FSM: Start Passive`。初始为阻尼状态。
5. 按 **L2 + 方向键上**进入 `FixStand`。程序用 3 秒过渡到训练默认站姿。保持吊装保护，让双脚稳定接触平整地面；悬空时腿部里程计的接触假设不成立。
6. 站姿稳定后，**持续按住 L2 至少 2 秒，再按方向键右**，进入 `BeyondMimic`。程序检查初始关节与参考姿态的最大偏差不超过 0.25 rad，然后按 50 Hz 推理。
7. 首次只运行默认 10 秒试验段。结束时会自动进入 **Passive 阻尼**；需要吊装承重。
8. 任何时候按 **L2 + B**退出策略。`Ctrl+C`/`SIGTERM` 会请求状态机发布阻尼再退出进程。断电、强制结束进程等情况不依赖这条正常退出路径。

本配置不包含速度行走模式；摇杆不会改变此模型的参考动作，也不能用官方通用行走示例的 `R1 + X` 来启动它。

## 6. 状态估计与保护的实际边界

机载编码器给出关节状态，IMU 给出骨盆姿态和角速度；位置和线速度使用腿部运动学估计。支撑脚在平地不滑动是估计成立的条件。支撑脚切换、打滑、腾空和 IMU 航向漂移都会影响结果，不能把这种估计视为绝对定位或完整状态估计器。

已加入的退出条件包括：LowState 超过 100 ms 未更新、推理结果超过 100 ms 未更新、策略错过周期、非有限传感量/动作、骨盆倾斜超过 0.85 rad、估计的躯干与参考位置误差超过 0.35 m，以及运行段结束。参考动作双脚明显离地时也退出。目标角受关节限位、变化速度和训练执行器力矩参数约束。

这些条件只覆盖代码能检测的异常；不替代真机的急停、吊装和人工监护。部署侧限幅本身也可能改变已学策略的行为，需要在训练和验证中同时考虑。

## 7. 已完成的验证与未完成项

本次实际验证见 `docs/beyondmimic_validation.json`：

- 导出模型与检查点推理的 64 组对照通过，最大绝对误差约 `2.4e-6`。
- 实际 C++ 观测与 MuJoCo 独立运动学对照通过，最大绝对误差约 `1.3e-7`。
- 控制器全部 G1 源文件使用真实宇树 SDK 头文件通过 ARM64 Linux 目标文件编译。离线程序在本机完成编译、链接和运行；完整真机控制器的 Linux 链接、DDS 通信与时序仍需在机载电脑验证。
- 默认配置使用腿部里程计完成 10 秒闭环；位置估计误差最大约 7 mm，速度估计误差最大约 0.17 m/s。这些仅是该次理想仿真结果。
- 延长到 40 秒时，带部署限幅的版本约在 15 秒失稳；使用真实仿真状态的对照也未通过。进一步的仿真对照表明，去除目标限幅、使用真实状态时可完成 40 秒。因此需要继续核对并在训练中体现部署限幅，同时改进状态估计，才能验证整段动作。
- **尚未在实际 G1 上测试，不能宣称整段真机部署已通过。**

不要仅增加 `duration_s` 就将整段动作放到真机上。更换运行段后需要重新验证初始姿态、状态估计和闭环；当前默认配置保留为 10 秒。

## 8. 重新导出或更换模型

在有 PyTorch 的开发电脑操作，不需要机载电脑安装训练环境：

```bash
pip install -r tools/requirements-beyondmimic.txt
python tools/export_beyondmimic.py \
  --training-root /实际路径/Beyondmimic_Weilai_G1 \
  --checkpoint /实际路径/Beyondmimic_Weilai_G1/trained_models/dance_zui_model_4999.pt \
  --motion /实际路径/Beyondmimic_Weilai_G1/whole_body_tracking/motions/dance_zui.npz \
  --output deploy/robots/g1_29dof/config/policy/beyondmimic/dance_zui
```

导出器读取训练源码中的关节顺序、默认姿态、执行器参数和 URDF 运动学。当前明确支持这个 160 维观测、29 维动作、50 Hz、ELU MLP 任务；154 维无状态估计模型、RNN 或其他机器人模型会被拒绝或需要另行适配。检查点的默认经验归一化使用 `(obs - mean) / (std + 0.01)`；若原训练修改了 epsilon，需要显式传入匹配的 `--normalizer-eps`。

参考动作必须与检查点对应，不应只因关节维度一致就替换成另一段动作。导出器的 shape 检查和散列记录不能证明任意检查点与动作的训练对应关系。

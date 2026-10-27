# rk3568_face_recognition

面向RK3568的Qt/C++人脸采集、检测与识别工程。使用OpenCV完成图像处理，支持RKNN/NPU检测和MobileFaceNet特征提取，并保留OpenCV/SFace与LBPH后端。

系统提供密码解锁值守界面与局域网网页管理。人脸采集、检测、跟踪和识别均由RK3568本地单路摄像头完成；电脑浏览器只管理人员库、识别记录和异常。详细验证状态见[验证记录](docs/access-control-validation.md)。

## 功能

- 板端默认仅显示摄像头预览和识别结果，点击“管理”后用独立密码进入原管理界面；首次在板卡设密，闲置5分钟返回值守。
- 网页管理支持人员与样本、识别记录、进出配对、异常处理和CSV导出。板端单路摄像头方向可设为进或出；一台摄像头不能同时代表两个物理出入口。
- 进入/离开不匹配会标记异常，未离开默认24小时超时，可修改或关闭；原记录保留当时姓名与授权快照。
- 人脸采集默认10张，可调整为10至500张；采集过程包含质量筛选和重复样本过滤。
- 程序默认打开配置中的摄像头，并在所选后端存在有效图库时开启实时识别。维护模式使用`--no-auto-start`。
- 人脸库管理支持按ID或姓名搜索、修改姓名与组别、启用/禁用、查看照片与样本数量，以及删除人员或单张样本。
- 删除会先保存人员表和样本索引归档，原照片保留；删除或改变启用状态后需重建图库。兼容版本清单防止重启时回退到删除前的旧特征。
- 支持原生V4L2 DMA-BUF导出取流、异步采集/检测，以及MIPI屏的eglfs全屏显示和旋转。
- 删除确认前固定人员和样本ID，异步预览刷新不会改变删除目标或访问已释放的列表项。
- 稳定识别人脸后按时段问候启用人员，陌生人或禁用人员提示“无权限，请联系管理员”；按光流/跟踪轨迹去重，确认丢失后可重新播报。

## 依赖与模型

已验证环境为Ubuntu22.04/aarch64、C++17、CMake3.22、Qt5.15.3、OpenCV4.5.4和RKNN Runtime2.3.2。CPU开发环境也可构建和运行离线回归。

Ubuntu构建依赖：

```bash
sudo apt-get install build-essential cmake ninja-build pkg-config qtbase5-dev qt5-qmake libopencv-dev libopencv-contrib-dev
```

模型放在`models/`，来源、输入合同和校验值见[模型说明](models/README.md)。NPU构建还需准备[RKNN SDK依赖](third_party/rknn/README.md)。仓库只跟踪源码和依赖清单，运行数据由本机应用数据目录保存。

## 构建

RK3568原生构建，SDK路径按本机实际位置填写：

```bash
cmake --preset rk3568-native-release \
  -DFACEAPP_RKNN_INCLUDE_DIR=/path/to/rknn/include \
  -DFACEAPP_RKNN_LIBRARY=/lib/librknnrt.so
cmake --build --preset rk3568-native-release -j2
```

CPU开发构建：

```bash
cmake --preset rk3568-cpu-only
cmake --build --preset rk3568-cpu-only -j2
```

根目录CMake构建应用和核心库，本机回归由`verification/`独立构建（含管理后台）。

## 管理后台环境

```bash
bash scripts/setup_access_server.sh
# 安装到项目server/.venv，不修改系统Python
```

管理服务默认监听所有网络接口的8080端口，由Qt通过子进程运行Flask/Waitress；`FACEAPP_ACCESS_PYTHON`可指定服务Python，`FACEAPP_ACCESS_PORT`可改端口。先在板卡设置管理密码，再在电脑浏览器访问`http://板卡IP:8080`管理数据库。网页管理支持上传人员样本照片，但不采集或上传实时视频供识别。运行目录、备份和回退见[运维说明](docs/access-control-operations.md)。

管理接口使用明文HTTP，密码、会话和人脸资料传输未加密。请仅在可信隔离的局域网使用；对外或生产部署需另行配置HTTPS和网络访问控制。系统尚未实现活体检测，不能据此保证抵御照片、屏幕或视频攻击。

## 启动

```bash
./build-rk3568/face_recognition_system --config config/rk3568.json
# 维护时先保留摄像头关闭
./build-rk3568/face_recognition_system --config config/rk3568.json --no-auto-start
```

MIPI屏使用：

```bash
bash scripts/run_on_mipi.sh "$PWD"
# 停止程序并恢复图形桌面
bash scripts/stop_on_mipi.sh "$PWD"
```

MIPI脚本涉及gdm3启停，需按板卡部署环境提供相应权限。默认画面旋转90度，可用`MIPI_ROTATION`调整。CPU开发构建可通过`FACEAPP_BUILD_DIR="$PWD/build-rk3568-cpu"`指定脚本使用的构建目录；在开发PC上选择可用的识别后端和视频设备。

运行数据默认位于Qt的`AppLocalDataLocation`，组织名为`PortableFaceSystem`、应用名为`FaceRecognitionSystem`。`FACEAPP_DATA_DIR=/path/to/runtime`可覆盖运行目录，配置中的相对`dataset_root`在该目录内解析；采集照片、人员表和生成图库保存在运行目录。

## 语音播报

固定提示音见[audio/README.md](audio/README.md)。默认走RK809扬声器，播放设备、启用开关和时段边界配置在`config/voice.json`；`FACEAPP_VOICE_CONFIG`可指定独立配置，`FACEAPP_VOICE_ENABLED=0`可关闭播报。

默认05:00—10:59播“早上好”，11:00—13:59播“中午好”，14:00—17:59播“下午好”，其余时间播“晚上好”，使用系统本地时间。组内/组外均按启用人员问候；陌生人和禁用人员播“无权限，请联系管理员”。低质量、等待投票、模型错误和导入静态图片不触发播报。

播报沿用现有光流预测和跟踪器ID：同一条存活轨迹只播一次，短暂遮挡保留状态，确认丢失或关闭输入后释放状态，重现并稳定识别后可再次播报。没有固定秒数冷却。播放使用异步`QProcess`，一次播放一段、最多排队8段，丢失或失去稳定识别的待播目标会被移出队列。播放失败只记日志，不中断识别。

桌面音频服务可用时优先通过`paplay`播放到RK809，Ubuntu系统音量和静音键生效。服务不存在或播放失败时回退`aplay`直连。全屏eglfs模式会停止GDM，Ubuntu桌面的音量快捷键服务也随之退出；此时可用界面的“播报音量”滑块、静音开关，或由程序处理键盘音量键。音量/静音保存到应用运行目录的`voice-preferences.ini`，再次启动恢复。界面音量控制RK809的DAC，桌面模式下还会叠加Ubuntu的软件音量。

显式测试真实音频可执行`FACEAPP_LIVE_AUDIO_TEST=1 ./build-verification/voice_tests liveRk809Playback`，会实际播放问候和拒绝提示。普通回归跳过该项。对独占显示环境，可在独立语音配置中设置`prefer_system_audio=false`验证ALSA通路。

## 测试

以下使用CPU构建的核心库；在板端可将路径替换为`build-rk3568`：

运行全部回归前，至少按[模型说明](models/README.md)准备`models/face_detection_yunet_2023mar.onnx`。后台适配器初始化使用这个检测模型；缺少时会明确报告依赖缺失。

```bash
cmake -S verification -B build-verification \
  -DCMAKE_BUILD_TYPE=Release -DFACEAPP_CORE_BUILD="$PWD/build-rk3568-cpu" \
  -DPython3_EXECUTABLE="$PWD/server/.venv/bin/python"
cmake --build build-verification -j2
ctest --test-dir build-verification --output-on-failure
```

测试覆盖主窗口默认参数与启动、管理窗口异步交互、数据库持久化、删除归档、图库失效及重启回退、身份缓存，以及语音时段、轨迹去重、过期队列与播放回退、音量/静音保存和音量键。实际摄像头用例需要显式设置`FACEAPP_LIVE_BOARD_TEST=1`并释放相机设备；普通离线测试会跳过该项。测试使用临时数据，正常回归不读取真实录入照片。

历次开发验证及适用边界见[验证记录](docs/access-control-validation.md)。测试与启动检查不等同于开放集识别准确率、活体防伪能力或持续帧率保证；DMA-BUF导出也不代表整条推理链路零拷贝。

## 目录

```text
src/app/              Qt线程适配、相机与识别业务
src/ui/               主窗口、人脸库管理和画面控件
src/core/             检测、跟踪、预处理、图库与数据存储
include/facecore/     核心库接口
config/               后端、相机、质量及阈值配置
verification/         9组本机回归、临时闭环与显式实机用例
audio/                固定播报WAV、生成记录与说明
server/               认证、SQLite、HTTP接口与管理网页
scripts/              构建、依赖、自检与板端启动入口
models/               模型清单和准备说明
third_party_licenses/ 上游模型许可文本
```

第三方依赖说明见[NOTICE.md](NOTICE.md)。

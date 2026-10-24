#!/usr/bin/env bash
# 板端最小编译依赖安装（ATK-DLRK3568 / Ubuntu 22.04 arm64）。
#
# 只安装构建与 GUI 所需的包，不做全系统升级：
#   sudo bash scripts/install_build_deps_rk3568.sh
#
# 实测结果（2026-09-22）：
#   g++ 11.4.0 / cmake 3.22.1 / ninja 1.10.1 / Qt 5.15.3 / OpenCV 4.5.4（含 face、objdetect）
#   安装耗时约 160 秒，占用约 0.9 GB。
#
# 注意：Ubuntu 22.04 自带的 OpenCV 是 4.5.4，只有 FaceDetectorYN/FaceRecognizerSF 的
# “文件路径”重载；本工程用 CMake 能力探测同时支持 4.5.4 与 4.7+ 两代 API，
# 因此不需要自行编译 OpenCV 4.11。

set -u
set -o pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "请用 sudo 运行：sudo bash $0" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
APT_OPTS=(-y -o Acquire::Retries=3 -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)

PACKAGES=(
    build-essential
    cmake
    ninja-build
    pkg-config
    git
    qtbase5-dev
    qt5-qmake
    libopencv-dev
    libopencv-contrib-dev
    libv4l-dev
    libgl1-mesa-dev
)

echo "== apt-get update =="
apt-get update || exit 1

echo "== apt-get install =="
apt-get install "${APT_OPTS[@]}" "${PACKAGES[@]}" || exit 1

echo
echo "== 校验 =="
for tool in gcc g++ make cmake ninja pkg-config qmake; do
    printf '%-12s %s\n' "$tool" "$(command -v "$tool" 2>/dev/null || echo MISSING)"
done
echo "opencv4      $(pkg-config --modversion opencv4 2>&1)"
echo "Qt5Widgets   $(pkg-config --modversion Qt5Widgets 2>&1)"
echo "libv4l2      $(pkg-config --modversion libv4l2 2>&1)"
echo "librknnrt    $(ls -l /lib/librknnrt.so 2>/dev/null || echo MISSING)"
echo "rknn driver  $(cat /sys/kernel/debug/rknpu/version 2>/dev/null || echo '需 root 或 debugfs')"
echo "OpenCV face  $(ls /usr/include/opencv4/opencv2/face.hpp 2>/dev/null || echo MISSING)"
echo "YuNet header $(ls /usr/include/opencv4/opencv2/objdetect/face.hpp 2>/dev/null || echo MISSING)"

#!/usr/bin/env bash
# 无界面自检：板端没有显示器、或通过 SSH 操作时使用。
#
#   bash scripts/self_test_rk3568.sh
#
# 依次检查：
#   1. 后端能力（RKNN / OpenCV DNN / LBPH）
#   2. librknnrt 与 RKNPU 驱动
#   3. 配置解析、检测后端、embedding 后端、摄像头候选
#   4. 摄像头只读探测（不打开设备）

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${FACEAPP_BUILD_DIR:-$ROOT/build-rk3568}"
CONFIG="${FACEAPP_CONFIG:-$ROOT/config/rk3568.json}"

if [ ! -x "$BUILD/face_recognition_system" ]; then
    echo "找不到可执行文件：$BUILD/face_recognition_system" >&2
    exit 1
fi

echo "=== 配置与后端自检 ==="
QT_QPA_PLATFORM=offscreen FACEAPP_CONFIG="$CONFIG" \
    "$BUILD/face_recognition_system" --self-test --config "$CONFIG"
exit_code=$?

echo
echo "=== 摄像头只读探测 ==="
if command -v v4l2-ctl >/dev/null 2>&1; then
    v4l2-ctl --list-devices 2>&1 | head -n 20
    if [ -e /dev/video-camera0 ]; then
        v4l2-ctl -d /dev/video-camera0 --get-fmt-video 2>&1 | head -n 6
    fi
else
    ls -l /dev/video-camera0 2>&1
fi

echo
echo "self-test exit=$exit_code"
exit "$exit_code"

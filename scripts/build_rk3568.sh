#!/usr/bin/env bash
# 板端原生 Release 构建 + 测试 + 自检。
#
#   bash scripts/build_rk3568.sh [源码根目录]
#
# 默认源码根目录是本脚本的上一级。构建目录为 <root>/build-rk3568。
# 不需要 root：librknnrt.so 是 0644，链接与运行都只需要读权限。

set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD="${FACEAPP_BUILD_DIR:-$ROOT/build-rk3568}"
CONFIG="${FACEAPP_CONFIG:-$ROOT/config/rk3568.json}"
VERIFY_BUILD="${FACEAPP_VERIFY_BUILD_DIR:-$ROOT/build-verification}"
JOBS="${FACEAPP_JOBS:-$(nproc)}"

cd "$ROOT" || exit 1
ACCESS_PYTHON="${FACEAPP_ACCESS_PYTHON:-$ROOT/server/.venv/bin/python}"
if [ ! -x "$ACCESS_PYTHON" ]; then
    echo "缺少管理服务Python环境，请先运行：bash $ROOT/scripts/setup_access_server.sh" >&2
    exit 1
fi
"$ACCESS_PYTHON" -I -c 'import flask, waitress, PIL, sqlite3' || exit 1

echo "== 环境 =="
echo "root    $ROOT"
echo "build   $BUILD"
echo "config  $CONFIG"
echo "g++     $(g++ --version | head -n 1)"
echo "cmake   $(cmake --version | head -n 1)"
if command -v ninja >/dev/null 2>&1; then echo "ninja   $(ninja --version)"; fi
echo "opencv4 $(pkg-config --modversion opencv4)"
echo "Qt5     $(pkg-config --modversion Qt5Widgets)"
echo "rknnrt  $(ls -l /lib/librknnrt.so 2>/dev/null || echo MISSING)"

echo
echo "== configure =="
cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFACEAPP_BUILD_GUI=ON \
    -DFACEAPP_ENABLE_RKNN="${FACEAPP_ENABLE_RKNN:-ON}" \
    -DFACEAPP_ENABLE_OPENCV_DNN="${FACEAPP_ENABLE_OPENCV_DNN:-ON}" \
    -DFACEAPP_ENABLE_LBPH="${FACEAPP_ENABLE_LBPH:-ON}" \
    -DFACEAPP_RKNN_INCLUDE_DIR="${FACEAPP_RKNN_INCLUDE_DIR:-$ROOT/third_party/rknn}" \
    -DFACEAPP_RKNN_LIBRARY="${FACEAPP_RKNN_LIBRARY:-/lib/librknnrt.so}" || exit 1

echo
echo "== build (-j$JOBS) =="
cmake --build "$BUILD" --parallel "$JOBS" || exit 1

echo
echo "== 动态库检查 =="
ldd "$BUILD/face_recognition_system" | grep -E 'rknn|opencv|Qt5|not found' || true

echo
echo "== ctest =="
cmake -S "$ROOT/verification" -B "$VERIFY_BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DFACEAPP_CORE_BUILD="$BUILD" \
    -DPython3_EXECUTABLE="$ACCESS_PYTHON"
cmake --build "$VERIFY_BUILD" --parallel "$JOBS"
ctest --test-dir "$VERIFY_BUILD" --output-on-failure


echo
echo "== 后端自检（offscreen，无 DISPLAY 也能跑）=="
QT_QPA_PLATFORM=offscreen FACEAPP_CONFIG="$CONFIG" \
    "$BUILD/face_recognition_system" --self-test --config "$CONFIG"

echo
echo "构建与自检完成。"

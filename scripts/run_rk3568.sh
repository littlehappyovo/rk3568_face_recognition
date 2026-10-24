#!/usr/bin/env bash
# 在 RK3568 本地桌面会话里启动 GUI。
#
#   bash scripts/run_rk3568.sh            # 用 config/rk3568.json
#   FACEAPP_CONFIG=config/default.json bash scripts/run_rk3568.sh
#
# 说明：
#   - SSH 会话里没有 DISPLAY，直接跑 GUI 会报 qt.qpa.plugin 错误。
#     这个脚本在检测不到 DISPLAY/WAYLAND_DISPLAY 时回落到 X11 :0，
#     但前提是你确实在本地控制台（或已 xhost 授权）登录过桌面。
#   - 只想做无界面自检请用 scripts/self_test_rk3568.sh。

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${FACEAPP_BUILD_DIR:-$ROOT/build-rk3568}"
CONFIG="${FACEAPP_CONFIG:-$ROOT/config/rk3568.json}"

if [ ! -x "$BUILD/face_recognition_system" ]; then
    echo "找不到可执行文件：$BUILD/face_recognition_system" >&2
    echo "先运行：bash $ROOT/scripts/build_rk3568.sh" >&2
    exit 1
fi

export FACEAPP_CONFIG="$CONFIG"

if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "[run_rk3568] 未检测到 DISPLAY/WAYLAND_DISPLAY，回落到 :0" >&2
    export DISPLAY=:0
fi

echo "[run_rk3568] config=$FACEAPP_CONFIG"
echo "[run_rk3568] DISPLAY=${DISPLAY:-<unset>} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<unset>}"
exec "$BUILD/face_recognition_system" "$@"
